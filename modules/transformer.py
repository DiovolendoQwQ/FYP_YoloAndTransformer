import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, channels, heads=8, dim_ff=None, dropout=0.0, residual_scale=1.0):
        super().__init__()
        dff = dim_ff if dim_ff is not None else channels * 2
        self.ln1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, dff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dff, channels)
        )
        self.residual_scale = residual_scale

    def forward(self, x):
        b, c, h, w = x.size()
        s = h * w
        t = x.view(b, c, s).permute(0, 2, 1)
        y = self.ln1(t)
        a, _ = self.attn(y, y, y)
        t = t + self.residual_scale * a
        y2 = self.ln2(t)
        f = self.ff(y2)
        t = t + self.residual_scale * f
        out = t.permute(0, 2, 1).view(b, c, h, w)
        return out

class C2fTransformerWrapper(nn.Module):
    def __init__(self, c2f_module, heads=8, dim_ff=None, dropout=0.0, min_size=64, residual_scale=1.0):
        super().__init__()
        self.c2f = c2f_module
        ch = getattr(getattr(c2f_module, 'cv2', None), 'conv', None)
        ch = getattr(ch, 'out_channels', None)
        self.tr = TransformerBlock(int(ch) if ch is not None else 1, heads=heads, dim_ff=dim_ff, dropout=dropout, residual_scale=residual_scale)
        self.min_size = min_size
        self.heads = heads
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.residual_scale = residual_scale
        for k in ('f', 'i', 'type', 'reparam', 'np', 'g', 'l'):
            if hasattr(c2f_module, k):
                setattr(self, k, getattr(c2f_module, k))

    def forward(self, x):
        y = self.c2f(x)
        if y.dim() == 4 and y.shape[-1] >= self.min_size and y.shape[-2] >= self.min_size:
            y = self.tr(y)
        return y
