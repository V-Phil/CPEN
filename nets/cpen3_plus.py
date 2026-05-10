import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from nets.Swin import SwinBackbone


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x

    # -----------------------------------------#


#   DACM（ASPP改）特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取

class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
   
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

   
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


# ------Edge Position Attention Module (EAM) ------
class PAM(nn.Module):
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1) # B, N, C'
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height) # B, C', N
        energy = torch.bmm(proj_query, proj_key) # B, N, N
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height) # B, C, N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, N
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM(nn.Module):
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key) # B, C, C
        # 防止数值溢出
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value) # B, C, N
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out




class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
       
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

        
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att 

       
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial_att))
        x = x * spatial_att 

        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class cpen(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(cpen, self).__init__()
        if backbone == "xception":
            from nets.xception import xception
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            from nets.mobilenetv2 import mobilenetv2
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "swin":
            self.backbone = SwinBackbone(model_name="swin_tiny_patch4_window7_224",
                                         pretrained=pretrained,
                                         img_size=512,
                                         out_stride=16)  # 16倍下采样。如果用out_stride=32用stage4和stage1
            in_channels = 384  # stage3输出 (如果你用out_stride=16)
            low_level_channels = 96  # stage1输出
        else:
            raise ValueError(f'Unsupported backbone - `{backbone}`, Use mobilenet, xception, swin.')

        # 后续结构保持不变
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        self.seblock = SEBlock(256)
        self.pam = PAM(256)
        self.cam = CAM(256)
        self.cbam = CBAM(in_channels=256)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        x = self.seblock(x)
        x = self.pam(x)
        x = self.cam(x)
        x = self.cbam(x)
        low_level_features = self.shortcut_conv(low_level_features)
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


