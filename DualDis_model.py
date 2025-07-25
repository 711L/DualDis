# -*- coding: utf-8 -*-
# @Author : wobhky
# @File : FilterModel.py
# @Time : 2025/4/28 20:41

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torchvision.models.resnet import resnet50, Bottleneck


class ACD(nn.Module):
    def __init__(self, num_embeddings=2048, num_heads=4):
        super(ACD, self).__init__()
        assert num_embeddings % num_heads == 0, "num_embeddings must be divisible by num_heads"
        self.embedding_dim = num_embeddings
        self.heads = num_heads
        self.head_dim = num_embeddings // num_heads

        # self.patch_opr = nn.Conv2d(3, num_embeddings,kernel_size=16,stride=16)#b,c,18,18
        self.q_proj = nn.ModuleList([
            nn.Linear(num_embeddings, self.head_dim) for _ in range(num_heads)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(num_embeddings, self.head_dim) for _ in range(num_heads)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(num_embeddings, num_embeddings) for _ in range(num_heads)
        ])  # The dimension is the feature dimension, hoping to see all the features.

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape  
        x_tmp = x
        patch_x = x.permute(0, 2, 3, 1).reshape(B, -1, self.embedding_dim) 
        Q = torch.stack([proj(patch_x) for proj in self.q_proj], dim=1)  
        K = torch.stack([proj(patch_x) for proj in self.k_proj], dim=1)  
        V = torch.stack([proj(patch_x) for proj in self.v_proj], dim=1)  
        similarity = torch.einsum('bhqd,bhkd->bhqk', Q, K) / (self.head_dim ** 0.5)  # b,4,324,324
        # rowi in similarity The attention of the i-th block to all blocks actually represents the value others have for you. 
        # The sum of each position in each row represents the importance of the j-th block to all blocks.
        attention = torch.sum(similarity, dim=2).unsqueeze(2) # b,4,1,324
        attention = F.softmax(attention, dim=1)
        _, _, _, patches = attention.shape
        # B,num_heads,18,18  attention_score 
        attention_score = attention.reshape(B, self.heads, int(patches ** 0.5),
                                            int(patches ** 0.5))  # It represents the importance of each block to other blocks under the current attention head.
        x_enhance = torch.concat([attention_score[:, i:i + 1, :, :] * x_tmp for i in range(self.heads)],
                                 1)  # b,4*num_embeddings,18,18
        similarity = F.softmax(similarity, dim=-1)
        general = similarity @ V  # b,4,324,512 -> B,4*512,18,18
        b, heads, len, dim = general.shape
        h, w = int(len ** 0.5), int(len ** 0.5)
        general = general.reshape(b, heads, h, w, dim)  # b,4,18,18,512
        general = general.permute(0, 1, 4, 2, 3).reshape(b, heads * dim, h, w)
        output = general + x_enhance
        return attention_score, output  # input->b,3,288,288 output->b,heads,18,18 and b,4*512,18,18


#Region-Aware  Sparse Channel Attention
class RASCA(nn.Module):
    def __init__(self, in_channels=2048, reduction_ratio=16, num_parts=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_parts = num_parts

        # Component segmentation ratio (top, middle, bottom)
        self.part_heights = [0.4, 0.3, 0.3]  # best chioce

        # Independent Channel Attention for Each Component
        self.part_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_parts)
        ])

        # Dynamic Component Weight Prediction (Automatically Assigning Weights Based on Input Features)
        self.part_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B,C,1,1]
            nn.Conv2d(in_channels, num_parts, 1),  # [B,3,1,1]
            nn.Softmax(dim=1)
        )

        # Channel Sparsity Control (Optional)
        self.sparsity_gate = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # ========== Part Segmentation ==========
        part_features = []
        start_h = 0
        for i in range(self.num_parts):
            part_h = int(h * self.part_heights[i])
            end_h = min(start_h + part_h, h)  # Prevent out-of-bounds
            part_feat = x[:, :, start_h:end_h, :]  # [B,C,part_h,W]
            part_features.append(part_feat)
            start_h = end_h

        # ========== Channel Attention ==========
        part_att_maps = []
        for i, part_feat in enumerate(part_features):
            # [B,C,1,1]
            att = self.part_attentions[i](part_feat.mean(dim=2, keepdim=True)) #b,1024,1,1
            part_att_maps.append(att)

        # ========== Dynamic Part Fusion ==========
        part_weights = self.part_weight(x)  # [B,3,1,1]
        fused_att = torch.zeros_like(part_att_maps[0]) #b,1024,1,1
        for i in range(self.num_parts):
            fused_att += part_weights[:, i:i + 1] * part_att_maps[i]

        # ==========  Channel Sparsification (Optional) ==========
        if hasattr(self, 'sparsity_gate'):
            sparsity = self.sparsity_gate(x.mean([2, 3]))  # [B,1]
            k = (sparsity * c).long().clamp(min=1, max=c)

            # Top-k selection
            channel_scores = fused_att.squeeze(-1).squeeze(-1)  # [B,C]
            mask = torch.zeros_like(channel_scores)
            for i in range(b):
                k_val = int(k[i].item())  # 将Tensor转为Python int
                topk_idx = channel_scores[i].topk(k_val)[1]
                mask[i, topk_idx] = 1
            fused_att = fused_att * mask.view(b, c, 1, 1)

        # ========== Final Output ==========
        return x * fused_att,x- x * fused_att, part_weights






# Symmetry-Aware Spatial Contextual  Attention SASCA
class SASCA(nn.Module):
    def __init__(self,in_channels = 2048):
        super().__init__()
        self.sym_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.asym_conv = nn.Conv2d(in_channels, in_channels // 2, 1)

        # Contextual 
        self.att_conv1 = nn.Conv2d(1, 1, kernel_size=7, padding=3)
        self.att_conv2 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.att_conv3 = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        # Weights
        self.weights = nn.Parameter(torch.ones(3) / 3)


    def forward(self, x):  # x->b,c,h,w
        # Symmetrical Feature Extraction
        x_flip = torch.flip(x, [3])
        sym_feat = (x + x_flip) / 2
        sym_feat = self.sym_conv(sym_feat)

        # Asymmetric Feature Extraction
        #asym_feat = torch.abs(x - x_flip)
        asym_feat = torch.relu(x - x_flip)
        asym_feat = self.asym_conv(asym_feat)

        diff = asym_feat.mean(1, keepdim=True)
        out1 = torch.sigmoid(self.att_conv1(diff))
        out2 = torch.sigmoid(self.att_conv2(diff))
        out3 = torch.sigmoid(self.att_conv3(diff))
        norm_weights = torch.softmax(self.weights, dim=0)
        attention = norm_weights[0] * out1 + norm_weights[1] * out2 + norm_weights[2] * out3

        # Fusion
        fused_feat = torch.cat([sym_feat * attention,asym_feat * (1 - attention)], dim=1)
        #fused_feat = torch.cat([sym_feat * (1 - attention), asym_feat * attention], dim=1)
        return fused_feat, x-fused_feat,attention




class PDA(nn.Module):
    def __init__(self):
        super().__init__()
        #print("VehicleAttention kwargs:", kwargs)
        self.rasca = RASCA()
        self.sasca = SASCA()

    def forward(self, x):
        out = x
        x,x_r1, weights = self.rasca(x)  

        x,x_r2,attention = self.sasca(x)

        return x+out, weights, attention, x_r1,x_r2,out-x


class ReductionFc(nn.Module):
    def __init__(self, feat_in, feat_out, num_classes):
        super(ReductionFc, self).__init__()

        self.reduction = nn.Sequential(nn.Conv2d(feat_in, feat_out, 1, bias=False),
                                       nn.BatchNorm2d(feat_out))  # , nn.ReLU()
        self._init_reduction(self.reduction)
        self.fc = nn.Linear(feat_out, num_classes, bias=False)
        self._init_fc(self.fc)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        # nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        reduce = self.reduction(x).view(x.size(0), -1)
        fc = self.fc(reduce)

        return reduce, fc


class DualDis_model(nn.Module):
    def __init__(self, fc_cls):
        super(DualDis_model, self).__init__()
        resnet_50 = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        self.post_backbone = nn.Sequential(
            resnet_50.conv1,
            resnet_50.bn1,
            resnet_50.relu,
            resnet_50.maxpool,
            resnet_50.layer1,
            resnet_50.layer2,
            resnet_50.layer3,
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, stride=1,
                       downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, stride=1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512), Bottleneck(2048, 512))
        self.up = ACD()
        self.down = PDA()

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.up_pool1 = nn.MaxPool2d((6, 18))#b,1024,3,1
        self.up_pool2 = nn.MaxPool2d((18, 6))#b,1024,1,3
        self.up_reduction1 = ReductionFc(2048*4, 256, fc_cls)
        self.up_reduction2 = ReductionFc(2048*4, 256, fc_cls)
        self.up_reduction3 = ReductionFc(2048*4, 256, fc_cls)
        self.up_reduction7 = ReductionFc(2048*4, 256, fc_cls)
        self.up_reduction8 = ReductionFc(2048*4, 256, fc_cls)
        self.up_reduction9 = ReductionFc(2048*4, 256, fc_cls)

        self.down_pool1 = nn.MaxPool2d((6, 18))  # b,1024,3,1
        self.down_pool2 = nn.MaxPool2d((18, 6))  # b,1024,1,3
        self.down_reduction1 = ReductionFc(2048, 256, fc_cls)
        self.down_reduction2 = ReductionFc(2048, 256, fc_cls)
        self.down_reduction3 = ReductionFc(2048, 256, fc_cls)
        self.down_reduction7 = ReductionFc(2048, 256, fc_cls)
        self.down_reduction8 = ReductionFc(2048, 256, fc_cls)
        self.down_reduction9 = ReductionFc(2048, 256, fc_cls)

        self.avgpool = nn.AvgPool2d((18, 18))
        self.mean_reduction = ReductionFc(2048, 256, fc_cls)
        self.global_reduction = ReductionFc(2048, 256, fc_cls)

        self.global1_reduction = ReductionFc(2048, 256, fc_cls)
        self.global2_reduction = ReductionFc(2048, 256, fc_cls)
        self.global3_reduction = ReductionFc(2048, 256, fc_cls)

        self.rub1 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.rub2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.r_reduction = ReductionFc(2048, 256, fc_cls)
        self.rub_maxpool = nn.MaxPool2d((18,18))
    def forward(self, x):
        x_backbone = self.post_backbone(x)  # b,2048,18,18
        x_backbone = self.layer4(x_backbone)  # b,2048,18,18


        # ACD
        score, extraction = self.up(x_backbone) 
        up_part1 = self.up_pool1(extraction)  
        up_part2 = self.up_pool2(extraction)  
        tri_up1, cls_up1 = self.up_reduction1(up_part1[:, :, 0:1, :])
        tri_up2, cls_up2 = self.up_reduction2(up_part1[:, :, 1:2, :])
        tri_up3, cls_up3 = self.up_reduction3(up_part1[:, :, 2:3, :])
        tri_up7, cls_up7 = self.up_reduction7(up_part2[:, :, :, 0:1])
        tri_up8, cls_up8 = self.up_reduction8(up_part2[:, :, :, 1:2])
        tri_up9, cls_up9 = self.up_reduction9(up_part2[:, :, :, 2:3])


        # PDA
        distilled, weights, attention,r1,r2,res = self.down(x_backbone)
        down_part1 = self.up_pool1(distilled) 
        down_part2 = self.up_pool2(distilled) 
        tri_down1, cls_down1 = self.down_reduction1(down_part1[:, :, 0:1, :])
        tri_down2, cls_down2 = self.down_reduction2(down_part1[:, :, 1:2, :])
        tri_down3, cls_down3 = self.down_reduction2(down_part1[:, :, 2:3, :])
        tri_down7, cls_down7 = self.down_reduction2(down_part2[:, :, :, 0:1])
        tri_down8, cls_down8 = self.down_reduction2(down_part2[:, :, :, 1:2])
        tri_down9, cls_down9 = self.down_reduction2(down_part2[:, :, :, 2:3])


        x_global1 = x_backbone * torch.max(score, 1, True)[0]
        tri_g1, cls_g1 = self.global1_reduction(self.maxpool(x_global1))
        part_heights = [0.4, 0.3, 0.3]
        b, c, h, w = x_backbone.shape
        # Generate Spatial Weight Mask (Based on part_heights ratio)
        spatial_mask = torch.zeros(b, 1, h, w, device=x.device)  # [B,1,H,W]
        start_h = 0
        for i in range(3):
            part_h = int(h * part_heights[i])
            end_h = min(start_h + part_h, h)
            spatial_mask[:, :, start_h:end_h, :] = weights[:, i:i + 1, :, :]  # 填充对应权重
            start_h = end_h

        weighted_x = x_global1 * spatial_mask.expand_as(x_backbone)
        x_global2 = weighted_x *  attention # attention：b，1,18,18
        tri_g2,cls_g2 = self.global2_reduction(self.maxpool(x_global2))

        predict = torch.concat([tri_up1,tri_up2,tri_up3,tri_up7,tri_up8,tri_up9,
                                tri_down1,tri_down2,tri_down3,tri_down7,tri_down8,tri_down9,
                                tri_g1,tri_g2], dim=1)


        return ([tri_g1,tri_g2,tri_up1,tri_up2,tri_up3,tri_up7,tri_up8,tri_up9,tri_down1,tri_down2,tri_down3,tri_down7,tri_down8,tri_down9],
                [cls_up1,cls_up2,cls_up3,cls_up7,cls_up8,cls_up9,
                cls_down1,cls_down2,cls_down3,cls_down7,cls_down8,cls_down9,cls_g1,cls_g2], predict)







