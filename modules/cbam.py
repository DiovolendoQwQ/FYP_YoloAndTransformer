import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, residual_scale=0.0, alpha=0.02):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.residual_scale = residual_scale
        self.alpha = alpha

    def forward(self, x):
        b, c, h, w = x.size()
        avg = torch.mean(x, dim=(2, 3), keepdim=False)
        maxv, _ = torch.max(x.view(b, c, -1), dim=2)
        mc = torch.sigmoid(self.mlp(avg) + self.mlp(maxv)).view(b, c, 1, 1)
        xs = torch.cat((torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]), dim=1)
        ms = torch.sigmoid(self.spatial(xs))
        att = (1.0 - self.alpha + self.alpha * mc) * (1.0 - self.alpha + self.alpha * ms)
        out = x * att
        if self.residual_scale > 0.0:
            return x + self.residual_scale * (out - x)
        return out

class C2fCBAMWrapper(nn.Module):
    def __init__(self, c2f_module, min_size=64, reduction=16, residual_scale=0.0):
        super().__init__()
        self.c2f = c2f_module
        ch = getattr(getattr(c2f_module, 'cv2', None), 'conv', None)
        ch = getattr(ch, 'out_channels', None)
        self.cbam = CBAM(int(ch) if ch is not None else 1, reduction, residual_scale=residual_scale)
        self.min_size = min_size
        self.reduction = reduction
        self.residual_scale = residual_scale
        for k in ('f', 'i', 'type', 'reparam', 'np', 'g', 'l'):
            if hasattr(c2f_module, k):
                setattr(self, k, getattr(c2f_module, k))

    def forward(self, x):
        y = self.c2f(x)
        if y.dim() == 4 and y.shape[-1] >= self.min_size and y.shape[-2] >= self.min_size:
            y = self.cbam(y)
        return y
