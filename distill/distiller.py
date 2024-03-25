

import torch
import torch.nn as nn

from trial.loss import sofTmax

epsilon = 1e-7


class FitNetDistiller(nn.Module):
    def __init__(self, student_model, teacher_model,
                 student_channel, teacher_channel, out_channel=256, distill_epoch=2,
                 T_weight=0.4, T_att=0.5,
                 weight_gl=0.1, weight_mhl=0.01, weight_atl=0.1,
                 max_error_num=16, high_iou_thresh=0.6, low_iou_thresh=0.1, low_score_thresh=0.3):
        super(FitNetDistiller, self).__init__()
        self.train_epoch = 0
        self.validate = False

        self.student_model = student_model
        self.teacher_model = teacher_model

        self.s_adp = nn.Sequential(nn.Conv2d(in_channels=student_channel, out_channels=out_channel, kernel_size=1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU())
        self.t_adp = nn.Sequential(nn.Conv2d(in_channels=teacher_channel, out_channels=out_channel, kernel_size=1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU())

        self.distill_epoch = distill_epoch
        self.T_weight, self.T_att = T_weight, T_att
        self.weight_gl, self.weight_mhl, self.weight_atl = weight_gl, weight_mhl, weight_atl
        self.max_error_num, self.high_iou_thresh, self.low_iou_thresh, self.low_score_thresh = \
            max_error_num, high_iou_thresh, low_iou_thresh, low_score_thresh

    def forward(self, data):
        if (self.train_epoch > self.distill_epoch) and (not self.validate):
            return self.forward_distill(data)
        else:
            return self.student_model.forward(data)

    def forward_distill(self, data):
        # 训练脚本编写, 设置文件编写
        positive = data['pos'].cuda().type(torch.float32)[:, None, None]
        label_cls = data['label_cls'].cuda()

        so = self.student_model.forward_student(data)
        to = self.teacher_model.forward_teacher(data)

        s_xf = so['f']
        b, c, h, w = s_xf.shape
        s_score = so['score']
        s_iou = so['iou']
        t_xf = to['f']
        t_score = to['score']
        t_iou = to['iou']

        with torch.no_grad():
            # 正负样本对两种情况未考虑
            # 得分图引导蒸馏
            # 负样本对上的iou值置0
            s_iou = s_iou * positive
            t_iou = t_iou * positive

            # strict teacher knowledge distillation, decouple target and background
            # 当教师模型在背景区域没有错误, 而学生模型在背景区域有错误结果时, 对该背景区域进行重点蒸馏, 且得分值将作为惩罚
            # 当教师模型在背景区域有任何错误时, 取消蒸馏; 当学生模型也没有错误时, 同样取消蒸馏
            neg_mask = (label_cls <= 0.).type(torch.float32)
            pos_mask = (label_cls > 0.).type(torch.float32)
            s_neg_iou_mask = (s_iou < self.low_iou_thresh).type(torch.float32)
            t_neg_iou_mask = (t_iou < self.low_iou_thresh).type(torch.float32)
            s_score_mask = (s_score > self.low_score_thresh).type(torch.float32)
            t_score_mask = (t_score > self.low_score_thresh).type(torch.float32)

            t_hard_neg = neg_mask * t_neg_iou_mask * t_score_mask
            t_error_num = t_hard_neg.sum(dim=(1, 2), keepdim=True)
            valid_teacher_background = (t_error_num < self.max_error_num).type(torch.float32)

            s_hard_neg = neg_mask * s_neg_iou_mask * s_score_mask * valid_teacher_background
            s_error_num = s_hard_neg.sum(dim=(1, 2), keepdim=True)
            hard_neg_num = (s_error_num > 0).type(torch.float32).sum() + epsilon

            s_hard_neg_score = s_score * s_hard_neg
            s_background_weight = sofTmax(s_hard_neg_score, T=self.T_weight, mask=s_hard_neg, average='batch')[:, None, ...]

            # 对于目标区域, 只有教师平均IoU大于学生平均IoU且高于IoU阈值时才蒸馏, 且IoU差值作为权重
            s_pos_iou = pos_mask * valid_teacher_background * (s_iou > self.high_iou_thresh).type(torch.float32) * s_iou
            t_pos_iou = pos_mask * valid_teacher_background * (t_iou > self.high_iou_thresh).type(torch.float32) * t_iou
            s_pos_iou_mask = (s_pos_iou > self.high_iou_thresh).type(torch.float32)
            t_pos_iou_mask = (t_pos_iou > self.high_iou_thresh).type(torch.float32)
            s_pos_iou_mean = s_pos_iou.sum(dim=(1, 2), keepdim=True) / (s_pos_iou_mask.sum(dim=(1, 2), keepdim=True) + epsilon)
            t_pos_iou_mean = t_pos_iou.sum(dim=(1, 2), keepdim=True) / (t_pos_iou_mask.sum(dim=(1, 2), keepdim=True) + epsilon)
            valid_teacher_foreground = (t_pos_iou_mean > s_pos_iou_mean).type(torch.float32)
            t_pos_mask = t_pos_iou_mask * valid_teacher_foreground
            pos_num = valid_teacher_foreground.sum() + epsilon

            t_pos_iou = t_iou * t_pos_mask
            iou_diff = t_pos_iou - (s_iou * t_pos_mask)
            t_foreground_weight = sofTmax(iou_diff, T=self.T_weight, mask=t_pos_mask, average='batch')[:, None, ...]

            valid_teacher_mask = (valid_teacher_background * valid_teacher_foreground)[..., None]
            valid_num = valid_teacher_mask.sum() + epsilon

        s_xf = self.s_adp(s_xf)
        t_xf = self.t_adp(t_xf)
        l2_loss = torch.nn.MSELoss(reduction='none')(s_xf, t_xf)

        background_loss = (l2_loss * s_background_weight).sum().div(hard_neg_num * c)
        foreground_loss = (l2_loss * t_foreground_weight).sum().div(pos_num * c)
        guided_loss = background_loss + foreground_loss

        s_att_ch = s_xf.mean(dim=1, keepdim=True)
        s_att_sp = s_xf.mean(dim=(2, 3), keepdim=True)
        t_att_ch = t_xf.mean(dim=1, keepdim=True)
        t_att_sp = t_xf.mean(dim=(2, 3), keepdim=True)

        with torch.no_grad():
            ch_mask = sofTmax(s_att_ch + t_att_ch, T=self.T_att, average='batch') * c
            sp_mask = sofTmax(s_att_sp + t_att_sp, T=self.T_att, average='batch') * (h * w)
        masked_hint_loss = torch.nn.MSELoss(reduction='none')(s_xf * ch_mask * sp_mask * valid_teacher_mask,
                                                              t_xf * ch_mask * sp_mask * valid_teacher_mask)
        masked_hint_loss = masked_hint_loss.sum().sqrt().div(valid_num * h * w * c)

        sp_l2_loss = torch.nn.MSELoss(reduction='none')(s_att_sp, t_att_sp) * valid_teacher_mask
        ch_l2_loss = torch.nn.MSELoss(reduction='none')(s_att_ch, t_att_ch) * valid_teacher_mask
        att_loss = sp_l2_loss.sum().div(valid_num * c) + ch_l2_loss.sum().div(valid_num * h * w)

        so['total_loss'] += self.weight_gl * guided_loss + self.weight_mhl * masked_hint_loss + self.weight_atl * att_loss
        so['guided_loss'] = guided_loss
        so['masked_hint_loss'] = masked_hint_loss
        so['att_loss'] = att_loss
        so.pop('f')
        so.pop('iou')
        so.pop('score')
        return so
