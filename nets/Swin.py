import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinBackbone(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True, img_size=512, out_stride=32):
        super(SwinBackbone, self).__init__()
        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       features_only=True,
                                       out_indices=(0, 1, 2, 3),
                                       img_size=img_size)
        self.out_stride = out_stride

    def forward(self, x):
        features = self.model(x)
        # 假设out_stride==16
        low_level = features[0]  # e.g. shape (B, H, W, C)
        out = features[2]  # e.g. shape (B, H, W, C)

        # 关键：变换shape
        low_level = low_level.permute(0, 3, 1, 2).contiguous()
        out = out.permute(0, 3, 1, 2).contiguous()

        return low_level, out

