# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import torch.nn as nn
from mmcv.ops.roi_align import RoIAlign

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
    def __init__(self, num_conv, in_channels, out_channels):
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
            RepBlock(out_channels * 2, out_channels, kernel_size=3, padding=1),
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
        output = torch.bmm(xf, zf)  # (b, 1024, 256)

        output = output.permute(0, 2, 1).contiguous().view(b, -1, h, w)  # (b, C, 32, 32)
        output = torch.cat((output, xf_), dim=1)  # (b, 2 * C, 32, 32)
        output = self.down_s(output)  # (b, C, 32, 32)

        bbox_tower = self.bbox_tower(output)
        bbox_reg = torch.exp(self.bbox_pred(bbox_tower))
        cls_tower = self.cls_tower(output)
        logits = self.cls_logits(cls_tower)

        f_ = torch.cat((output, cls_tower), dim=1).detach()
        return logits, bbox_reg, f_, xf_


class MobileSiamBuilder(BaseModelBuilder):
    def __init__(self, cfg):
        super(MobileSiamBuilder, self).__init__(cfg)

        # process zf
        self.process_zf = Process_zf(in_channels=cfg.BACKBONE.OUTPUT_CHANNELS, out_channels=cfg.TRAIN.NUM_CHANNELS)

        # build head
        self.head = Head_conf(num_conv=cfg.TRAIN.NUM_CONVS,
                              in_channels=cfg.BACKBONE.OUTPUT_CHANNELS, out_channels=cfg.TRAIN.NUM_CHANNELS)

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
        head_params = [self.head.parameters(), self.process_zf.parameters()]
        # head_params = [self.head.parameters(), self.process_zf.parameters(), self.process_conf.parameters()]
        return head_params

    def template(self, z):
        z = normalize_batch_cuda(z, self.mean, self.std, False)
        zf = self.backbone(z)
        zf = self.process_zf(zf)
        self.zf = zf

    def track(self, x):
        x = normalize_batch_cuda(x, self.mean, self.std, False)
        xf = self.backbone(x)
        cls, loc, f_ = self.head(self.zf, xf)[:3]
        # conf = self.process_conf(f_)

        return {
                'cls': cls,
                'loc': loc,
                # 'conf': conf
               }

    def forward_param(self, x):
        # z = torch.randn((1, 3, 128, 128))
        # zf = self.backbone(z)
        # zf = self.process_zf(zf)

        # zf = torch.randn((1, 256, 256))
        # tf = self.updater(zf, zf)

        zf = torch.randn((1, 256, 256))
        xf = self.backbone(x)
        cls, loc, f_ = self.head(zf, xf)[:3]
        return f_

    def forward_teacher(self, data):
        """ only used in training
        """
        with torch.no_grad():
            template_ = data['template'].cuda()
            search_ = data['search'].cuda()
            bbox = data['bbox'].cuda()[:, None, None, :]

            template = normalize_batch_cuda(template_, self.mean, self.std, False)
            search = normalize_batch_cuda(search_, self.mean, self.std, False)

            # get feature
            zf = self.backbone(template)
            xf = self.backbone(search)
            zf = self.process_zf(zf)
            cls_, loc, f_, xf = self.head(zf, xf)

            loc = torch.clamp(loc, max=1e4)
            loc = torch.where(torch.isinf(loc), 1e4 * torch.ones_like(loc), loc)

            score, boxes = LTRBDecoder(cls_, loc, self.points, self.cfg.TRAIN.SEARCH_SIZE)
            boxes = boxes.permute(0, 2, 3, 1).contiguous()
            bbox = process_box(bbox)
            boxes = process_box(boxes)
            iou, union_area = bbox_iou(bbox, boxes)

            outputs = {}
            outputs['f'] = xf
            outputs['score'] = score
            outputs['iou'] = iou
            # outputs['boxes'] = boxes

            # a0 = data['label_cls'].numpy()
            # a1 = iou.cpu().detach().numpy()
            # a2 = score.cpu().detach().numpy()
            # a3 = neg_weight.cpu().detach().numpy()
            # import cv2
            # search_img = search_.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
            # template_img = template_.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
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
            # cv2.waitKey()
            return outputs

    def forward_student(self, data):
        template_ = data['template'].cuda()
        search_ = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, :]
        positive = data['pos'].cuda().type(torch.float32)

        template = normalize_batch_cuda(template_, self.mean, self.std, False)
        search = normalize_batch_cuda(search_, self.mean, self.std, False)

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        zf = self.process_zf(zf)
        cls_, loc, f_, xf = self.head(zf, xf)

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
                pos_weight_cls = sofTmax(iou, T=self.T, b=(1 - self.ti), mask=pos_mask, average='batch')
                pos_weight_l1 = sofTmax(score, T=self.T, b=(1 - self.ts), mask=pos_mask, average='batch')
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

        outputs = {}
        outputs['f'] = xf
        outputs['score'] = score
        outputs['iou'] = iou
        # outputs['boxes'] = boxes
        outputs['total_loss'] = cls_loss + loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['pos_loss'] = pos_loss
        outputs['neg_loss'] = neg_loss
        outputs['loc_loss'] = loc_loss
        outputs['iou_loss'] = iou_loss
        outputs['l1_loss'] = l1_loss
        return outputs

    def forward_original(self, data):
        return 0

    def forward_trial(self, data):
        """ only used in training
        """
        template_ = data['template'].cuda()
        search_ = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, :]
        positive = data['pos'].cuda().type(torch.float32)

        template = normalize_batch_cuda(template_, self.mean, self.std, False)
        search = normalize_batch_cuda(search_, self.mean, self.std, False)

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        zf = self.process_zf(zf)
        cls_, loc, f_ = self.head(zf, xf)[:3]

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
                pos_weight_cls = sofTmax(iou, T=self.T, b=(1 - self.ti), mask=pos_mask, average='batch')
                pos_weight_l1 = sofTmax(score, T=self.T, b=(1 - self.ts), mask=pos_mask, average='batch')
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
        # cv2.waitKey()

        outputs = {}
        outputs['total_loss'] = cls_loss + loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['pos_loss'] = pos_loss
        outputs['neg_loss'] = neg_loss
        outputs['loc_loss'] = loc_loss
        outputs['iou_loss'] = iou_loss
        outputs['l1_loss'] = l1_loss
        return outputs


if __name__ == '__main__':
    pass
