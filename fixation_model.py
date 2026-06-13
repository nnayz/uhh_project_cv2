
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def get_center_bias():
    return torch.from_numpy(np.load(r'.\cv2_project_data\center_bias_density.npy')).float()


class FixationNet(nn.Module):
    def __init__(self, window_size=25, sigma=11.2):
        super().__init__()
        self.fcn = fcn_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=1
        )

        g = gaussian(window_size, sigma)
        kernel = torch.matmul(g.unsqueeze(-1), g.unsqueeze(-1).t())
        self.smooth_kernel = nn.Parameter(data=kernel, requires_grad=False)
        self.log_center_bias = nn.Parameter(data=torch.log(get_center_bias()), requires_grad=False)
        # for param in self.fcn.backbone.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        res = self.fcn(x)
        out = res['out']
        smoothed = F.conv2d(
            out,
            self.smooth_kernel.unsqueeze(0).unsqueeze(0),
            padding="same"
        )
        center_fix = smoothed + self.log_center_bias
        return center_fix
