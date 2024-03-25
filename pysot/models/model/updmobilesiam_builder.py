# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import torch.nn as nn

from pysot.models.model.base_model_builder import BaseModelBuilder
from pysot.models.backbone.repvgg import RepVGGBlock as RepBlock
from trial.loss import label2weight, label_update_, sofTmax
from trial.loss import weighted_select_cross_entropy_loss, weighted_iou_loss, weighted_l1_loss
from trial.utils.iou import bbox_iou, process_box
from trial.Decoders import LTRBDecoder
from trial.utils.image import normalize_batch_cuda


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, x):
        b, c, w, h = x.shape
        x1 = x.view((b, c, -1))
        x2 = x1.permute(0, 2, 1).contiguous()
        x_ = torch.bmm(x2, x1)
        x_ = torch.softmax(x_, -1)
        out = torch.bmm(x1, x_)
        return out


class Process_zf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Process_zf, self).__init__()
        self.out_channels = out_channels
        self.down_fz = nn.Sequential(
            RepBlock(in_channels, out_channels, kernel_size=3, padding=1),
            RepBlock(out_channels, out_channels, kernel_size=3, padding=1),
            RepBlock(out_channels, out_channels, kernel_size=3, padding=1))
        self.self_attention = SelfAttention()

    def forward(self, zf):
        zf = torch.cat(zf, dim=1)
        zf = self.down_fz(zf)  # (4, 256, 16, 16)
        # zf = self.cbam(zf)
        zf = self.self_attention(zf)  # (4, 256, 256)
        return zf


class Head_conf(nn.Module):
    def __init__(self, num_conv, nz, in_channels, out_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(Head_conf, self).__init__()

        self.down_fx = nn.Sequential(
            RepBlock(in_channels, out_channels, kernel_size=3, padding=1),
            RepBlock(out_channels, out_channels, kernel_size=3, padding=1),
            RepBlock(out_channels, out_channels, kernel_size=3, padding=1))
        self.down_s = nn.Sequential(
            RepBlock(out_channels + nz * 2, out_channels, kernel_size=3, padding=1),
            RepBlock(out_channels, out_channels, kernel_size=3, padding=1),
            RepBlock(out_channels, out_channels, kernel_size=3, padding=1))

        cls_tower = []
        bbox_tower = []
        for i in range(num_conv):
            cls_tower.append(RepBlock(out_channels, out_channels, kernel_size=3, padding=1))
            bbox_tower.append(RepBlock(out_channels, out_channels, kernel_size=3, padding=1))

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(out_channels, 2, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(out_channels, 4, kernel_size=3, stride=1, padding=1)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
                    torch.nn.init.normal_(l.weight, std=0.01)

    def forward(self, zf, xf):
        b, c, h, w = xf[0].shape
        xf = torch.cat(xf, dim=1)
        xf_ = self.down_fx(xf)  # (b, C, 32, 32)

        xf = xf_.flatten(2).permute(0, 2, 1).contiguous()  # (b, 1024, C)
        output = torch.bmm(xf, zf)  # (b, 1024, 256 * 2)

        output = output.permute(0, 2, 1).contiguous().view(b, -1, h, w)  # (b, 256 * 2, 32, 32)
        output_ = torch.cat((output, xf_), dim=1)  # (b, 256 * 2 + C, 32, 32)
        output = self.down_s(output_)  # (b, 256, 32, 32)

        bbox_tower = self.bbox_tower(output)
        bbox_reg = torch.exp(self.bbox_pred(bbox_tower))
        cls_tower = self.cls_tower(output)
        logits = self.cls_logits(cls_tower)

        f_ = torch.cat((output_, logits), dim=1).detach()
        # f_ = logits.detach()
        return logits, bbox_reg, f_


class Process_conf(nn.Module):
    def __init__(self, in_channels):
        super(Process_conf, self).__init__()

        self.down_conf = nn.Sequential(
            RepBlock(770, in_channels, kernel_size=3, padding=1),
            RepBlock(in_channels, in_channels, kernel_size=3, padding=1),
            RepBlock(in_channels, 4, kernel_size=3, padding=1))
        self.conf_fc1 = nn.Linear(in_features=4096, out_features=512)
        self.conf_fc2 = nn.Linear(in_features=512, out_features=1)
        self.conf_act = nn.ReLU()

        # initialization
        for modules in [self.down_conf, self.conf_fc1, self.conf_fc2]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
                    torch.nn.init.normal_(l.weight, std=0.01)

    def forward(self, x):
        conf = self.down_conf(x)
        conf = conf.flatten(1)
        conf = self.conf_fc2(self.conf_act(self.conf_fc1(conf))).sigmoid().squeeze()
        return conf


class TempUpdater(nn.Module):
    def __init__(self, in_channels):
        super(TempUpdater, self).__init__()

        self.fc1 = nn.Linear(in_features=in_channels, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=in_channels)
        self.act = nn.ReLU()
        self.factor = 8. if math.sqrt(in_channels) > 10. else math.sqrt(in_channels)
        # initialization
        for modules in [self.fc1, self.fc2]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
                    torch.nn.init.normal_(l.weight, std=0.01)

    def forward(self, t0, t1):
        # t0: (b, d, 256)
        # t1: (b, d, T*256)

        t0_ = t0.permute(0, 2, 1).contiguous()  # (b, 256, d)
        t1_ = t1.permute(0, 2, 1).contiguous()  # (b, T*256, d)
        att = torch.bmm(t1_, t0).div(self.factor).softmax(dim=-1)  # (b, T*256, 256)

        t1_ = torch.bmm(att, t0_) + t1_  # (b, T*256, d)

        t = torch.cat((t0_, t1_), dim=1)  # (b, 256*(T+1), d)

        t = self.fc2(self.act(self.fc1(t))) + t

        t = t.permute(0, 2, 1).contiguous()  # (b, d, 256*(T+1))
        return t


class UPDMobileSiamBuilder(BaseModelBuilder):
    def __init__(self, cfg):
        super(UPDMobileSiamBuilder, self).__init__(cfg)

        self.nz = cfg.TRAIN.ZF_SIZE ** 2

        # process zf
        self.process_zf = Process_zf(in_channels=cfg.BACKBONE.OUTPUT_CHANNELS, out_channels=cfg.TRAIN.NUM_CHANNELS)

        # build head
        self.head = Head_conf(num_conv=cfg.TRAIN.NUM_CONVS, nz=cfg.TRAIN.ZF_SIZE ** 2,
                              in_channels=cfg.BACKBONE.OUTPUT_CHANNELS, out_channels=cfg.TRAIN.NUM_CHANNELS)

        self.updater = TempUpdater(in_channels=cfg.TRAIN.NUM_CHANNELS)

        self.process_conf = Process_conf(in_channels=cfg.TRAIN.NUM_CHANNELS)
        self.conf_loss_func = nn.BCELoss()

        # self._reset_parameters()
        self.mean = torch.tensor(cfg.NORMALIZE_MEAN, dtype=torch.float32).view((1, -1, 1, 1)).cuda()
        self.std = torch.tensor(cfg.NORMALIZE_STD, dtype=torch.float32).view((1, -1, 1, 1)).cuda()

    def _reset_parameters(self):
        for modules in [self.head]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform(l.weight)
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

    def get_head_parameters(self):
        head_params = [self.head.parameters(), self.process_zf.parameters(), self.updater.parameters()]
        # head_params.append(self.process_conf.parameters())
        return head_params

    def template(self, z):
        z = normalize_batch_cuda(z, self.mean, self.std, False)
        zf = self.backbone(z)
        zf = self.process_zf(zf)
        self.zf = zf
        self.tf = self.updater(zf, zf)

    def update(self, t):
        t = normalize_batch_cuda(t, self.mean, self.std, False)
        tf = self.backbone(t)
        tf = self.process_zf(tf)
        tf = self.updater(self.zf, tf)
        self.tf = tf

    def track(self, x):
        x = normalize_batch_cuda(x, self.mean, self.std, False)
        xf = self.backbone(x)
        cls, loc, f_ = self.head(self.tf, xf)
        conf = self.process_conf(f_)

        return {
                'cls': cls,
                'loc': loc,
                'conf': conf
               }

    def track_conf(self, x):
        x = normalize_batch_cuda(x, self.mean, self.std, False)
        xf = self.backbone(x)
        cls, loc, f_ = self.head(self.tf, xf)

        return {
                'cls': cls,
                'loc': loc,
                'f_': f_
               }

    def forward_param(self, x):
        # z = torch.randn((1, 3, 128, 128))
        # zf = self.backbone(z)
        # zf = self.process_zf(zf)

        zf = torch.randn((1, 256, 256))
        tf = self.updater(zf, zf)

        # tf = torch.randn((1, 256, 512))
        xf = self.backbone(x)
        cls, loc, f_ = self.head(tf, xf)
        # return f_
        conf = self.process_conf(f_)
        return conf

    def forward_original(self, data):
        return 0

    def forward_trial(self, data):
        # return self.forward_trial_norm(data)
        return self.forward_trial_conf(data)  # only train conf head

    def forward_trial_norm(self, data):
        """ only used in training
        """
        template_ = data['template'].cuda()
        search_ = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, :]
        positive = data['pos'].cuda().type(torch.float32)
        temp_imgs = data['temp_imgs'][0].cuda()
        temp_boxes = data['temp_boxes'][0]
        pos_temp = data['pos_temp'][0].cuda().type(torch.float32)

        pos_temp = pos_temp * positive
        pos_temp_ = (pos_temp == 0.).type(torch.float32)
        pos_temp_ = pos_temp_ * positive
        all_pos_rate = 1.6
        pos_num = positive.sum()
        all_pos_num = pos_temp.sum()
        only_z_pos_num = pos_temp_.sum()
        if all_pos_num > 0 and only_z_pos_num > 0:
            only_z_temp_weight = pos_num / (all_pos_rate * all_pos_num + only_z_pos_num)
            all_pos_temp_weight = all_pos_rate * only_z_temp_weight
            temp_weight = pos_temp * all_pos_temp_weight + pos_temp_ * only_z_temp_weight
            temp_weight = temp_weight.view((-1, 1, 1))
        else:
            temp_weight = positive.detach().view((-1, 1, 1))

        template = normalize_batch_cuda(template_, self.mean, self.std, False)
        search = normalize_batch_cuda(search_, self.mean, self.std, False)
        temp_imgs = normalize_batch_cuda(temp_imgs, self.mean, self.std, False)

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        zf = self.process_zf(zf)

        tf = self.backbone(temp_imgs)
        tf = self.process_zf(tf)
        zf = self.updater(zf, tf)

        cls_, loc, f_ = self.head(zf, xf)

        loc = torch.clamp(loc, max=1e4)
        loc = torch.where(torch.isinf(loc), 1e4 * torch.ones_like(loc), loc)

        cls = self.log_softmax(cls_)
        score, boxes = LTRBDecoder(cls_, loc, self.points, self.cfg.TRAIN.SEARCH_SIZE)
        boxes = boxes.permute(0, 2, 3, 1).contiguous()
        bbox = process_box(bbox)
        boxes = process_box(boxes)
        iou, union_area = bbox_iou(bbox, boxes)

        with torch.no_grad():
            # 基于self-training方式，根据pred boxes与GT直接的iou以及boxes的score，重新划分正负样本
            if self.update_settings is not None and not self.validate:
                label_cls = label_update_(label_cls.cpu().detach().numpy(),
                                          score.cpu().detach().numpy(), iou.cpu().detach().numpy(),
                                          positive.cpu().detach().numpy(), **self.update_settings)
            pos_mask = (label_cls == 1.).type(torch.float32)
            neg_mask = (label_cls == 0.).type(torch.float32)

            if 'weighted' in self.cfg.MODE and self.train_epoch > 0 and not self.validate:
                iou_weight = sofTmax(-iou * positive[:, None, None], T=self.T, b=-self.ti, mask=neg_mask, average='batch')
                neg_weight = sofTmax(score, T=self.T, b=self.ts, mask=neg_mask * iou_weight, average='batch')
                pos_weight_cls = sofTmax(iou, T=self.T, b=(1 - self.ti), mask=pos_mask, average='batch') * temp_weight
                pos_weight_l1 = sofTmax(score, T=self.T, b=(1 - self.ts), mask=pos_mask, average='batch') * temp_weight
            else:
                neg_weight = label2weight(neg_mask, avg='batch')
                pos_weight_cls = label2weight(pos_mask, avg='batch')
                pos_weight_l1 = pos_weight_cls
            pos_weight_iou = pos_weight_l1

        l1_loss = weighted_l1_loss(label_loc, loc, pos_weight_l1, smooth=True)

        # cross entropy loss
        pos_loss, neg_loss = weighted_select_cross_entropy_loss(cls, label_cls,
                                                                pos_weight=pos_weight_cls, neg_weight=neg_weight)
        cls_loss = pos_loss * self.weights['pos_weight'] + neg_loss * self.weights['neg_weight']

        iou_loss = weighted_iou_loss(bbox, boxes, weight=pos_weight_iou,
                                     iou=iou, union_area=union_area, loss_type='ciou')[0]

        loc_loss = iou_loss * self.weights['iou_weight']
        if self.train_epoch > 0 and self.weights['l1_weight']:
            loc_loss += l1_loss * self.weights['l1_weight']

        # a0 = data['label_cls'].numpy()
        # a1 = iou.cpu().detach().numpy()
        # a2 = score.cpu().detach().numpy()
        # a3 = neg_weight.cpu().detach().numpy()
        # import cv2
        # search_img = search_.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # template_img = template_.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # temp_imgs = temp_imgs.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # # j = 2
        # # cv2.imshow('0', template_img[j])
        # # box = list(map(int, data['bbox'].numpy()[j]))
        # # img = search_img[j]
        # # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # # cv2.imshow('1', img)
        # # cv2.waitKey()
        # for j in range(search_img.shape[0]):
        #     temp_img = cv2.cvtColor(template_img[j], cv2.COLOR_RGB2BGR)
        #     cv2.imshow(str(j) + '_temp', temp_img)
        #     box = list(map(int, data['bbox'].numpy()[j]))
        #     img = cv2.cvtColor(search_img[j], cv2.COLOR_RGB2BGR)
        #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #     cv2.imshow(str(j) + '_search', img)
        #     box = list(map(int, temp_boxes.numpy()[j]))
        #     temp_img_ = cv2.cvtColor(temp_imgs[j], cv2.COLOR_RGB2BGR)
        #     cv2.rectangle(temp_img_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #     cv2.imshow(str(j) + '_online_temp', temp_img_)
        # cv2.waitKey()

        outputs = {}
        outputs['total_loss'] = cls_loss + loc_loss  #+ conf_loss * 0.4
        # outputs['conf_loss'] = conf_loss
        outputs['cls_loss'] = cls_loss
        outputs['pos_loss'] = pos_loss
        outputs['neg_loss'] = neg_loss
        outputs['loc_loss'] = loc_loss
        outputs['iou_loss'] = iou_loss
        outputs['l1_loss'] = l1_loss
        return outputs

    def forward_trial_conf(self, data):
        """ only used in training
        """
        template_ = data['template'].cuda()
        search_ = data['search'].cuda()
        positive = data['pos'].cuda().type(torch.float32)
        temp_imgs = data['temp_imgs'][0].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, :]

        template = normalize_batch_cuda(template_, self.mean, self.std, False)
        search = normalize_batch_cuda(search_, self.mean, self.std, False)
        temp_imgs = normalize_batch_cuda(temp_imgs, self.mean, self.std, False)

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        zf = self.process_zf(zf)

        tf = self.backbone(temp_imgs)
        tf = self.process_zf(tf)
        zf = self.updater(zf, tf)

        cls_, loc, f_ = self.head(zf, xf)

        conf = self.process_conf(f_)
        conf_loss = self.conf_loss_func(conf, positive)

        with torch.no_grad():
            negative = (positive == 0.).type(torch.float32)
            pred_pos = (conf > 0.6).type(torch.float32)
            pred_neg = (conf < 0.3).type(torch.float32)
            true_pos = pred_pos * positive
            true_neg = pred_neg * negative
            true_pred = ((true_pos + true_neg) == 1.).type(torch.float32)
            true_num = true_pred.sum()
            pos_num = positive.sum()
            neg_num = negative.sum()
            tp = true_pos.sum() / (pos_num + 1e-6)
            tn = true_neg.sum() / (neg_num + 1e-6)

        loc = torch.clamp(loc, max=1e4)
        loc = torch.where(torch.isinf(loc), 1e4 * torch.ones_like(loc), loc)
        cls = self.log_softmax(cls_)
        score, boxes = LTRBDecoder(cls_, loc, self.points, self.cfg.TRAIN.SEARCH_SIZE)
        boxes = boxes.permute(0, 2, 3, 1).contiguous()
        bbox = process_box(bbox)
        boxes = process_box(boxes)
        iou, union_area = bbox_iou(bbox, boxes)

        with torch.no_grad():
            pos_mask = (label_cls == 1.).type(torch.float32)
            neg_mask = (label_cls == 0.).type(torch.float32)
            neg_weight = label2weight(neg_mask, avg='batch')
            pos_weight_cls = label2weight(pos_mask, avg='batch')
            pos_weight_l1 = pos_weight_cls
            pos_weight_iou = pos_weight_l1

        l1_loss = weighted_l1_loss(label_loc, loc, pos_weight_l1, smooth=True)
        iou_loss = weighted_iou_loss(bbox, boxes, weight=pos_weight_iou,
                                     iou=iou, union_area=union_area, loss_type='ciou')[0]
        # cross entropy loss
        pos_loss, neg_loss = weighted_select_cross_entropy_loss(cls, label_cls,
                                                                pos_weight=pos_weight_cls, neg_weight=neg_weight)
        # a0 = data['label_cls'].numpy()
        # a1 = iou.cpu().detach().numpy()
        # a2 = score.cpu().detach().numpy()
        # a3 = neg_weight.cpu().detach().numpy()
        # import cv2
        # search_img = search_.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # template_img = template_.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # temp_imgs = temp_imgs.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # # j = 2
        # # cv2.imshow('0', template_img[j])
        # # box = list(map(int, data['bbox'].numpy()[j]))
        # # img = search_img[j]
        # # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # # cv2.imshow('1', img)
        # # cv2.waitKey()
        # for j in range(search_img.shape[0]):
        #     temp_img = cv2.cvtColor(template_img[j], cv2.COLOR_RGB2BGR)
        #     cv2.imshow(str(j) + '_temp', temp_img)
        #     box = list(map(int, data['bbox'].numpy()[j]))
        #     img = cv2.cvtColor(search_img[j], cv2.COLOR_RGB2BGR)
        #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #     cv2.imshow(str(j) + '_search', img)
        #     box = list(map(int, temp_boxes.numpy()[j]))
        #     temp_img_ = cv2.cvtColor(temp_imgs[j], cv2.COLOR_RGB2BGR)
        #     cv2.rectangle(temp_img_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #     cv2.imshow(str(j) + '_online_temp', temp_img_)
        # cv2.waitKey()

        outputs = {}
        outputs['total_loss'] = conf_loss
        outputs['true_num'] = true_num
        outputs['TP'] = tp
        outputs['TN'] = tn
        outputs['pos_loss'] = pos_loss
        outputs['neg_loss'] = neg_loss
        outputs['iou_loss'] = iou_loss
        outputs['l1_loss'] = l1_loss
        return outputs


if __name__ == '__main__':
    pass
