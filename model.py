import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from einops import rearrange
import torch._dynamo as dynamo

dynamo.config.cache_size_limit = 64


##### pointwise function #####
def rmsnorm_function(x, scale, eps):
    norm = x.norm(2, dim=-1, keepdim=True)
    rms = norm / (x.size(-1) ** 0.5)
    return scale[None, :, None, None] * x / (rms + eps)


def pointwise_gating(x, act_fn):
    lin, gate = rearrange(x, "n (g c) h w -> g n c h w", g=2)
    return lin * act_fn(gate)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8, compile_fn=True):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))
        self.compile_fn = compile_fn

    def forward(self, x):
        if self.compile_fn:
            return torch.compile(rmsnorm_function(x, self.scale, self.eps))
        else:
            rmsnorm_function(x, self.scale, self.eps)


class TimestepRMSNorm(nn.Module):
    def __init__(self, t, d, eps=1e-8, compile_fn=True):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Linear(t, d, bias=False)
        self.compile_fn = compile_fn

    def forward(self, x, t):
        scale = self.scale(t)

        if self.compile_fn:
            return torch.compile(rmsnorm_function(x, scale, self.eps))
        else:
            rmsnorm_function(x, scale, self.eps)


class SimpleResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        timestep_embed_size=1,
        act_fn=torch.sin,
        compile_fn=True,
    ):
        super(SimpleResNetBlock, self).__init__()
        self.compile_fn = compile_fn
        self.timestep_embed_size = timestep_embed_size
        if timestep_embed_size:
            self.timestep_rms_norm = TimestepRMSNorm(
                timestep_embed_size, out_channels, compile_fn=compile_fn
            )
        else:
            self.rms_norm = RMSNorm(out_channels, compile_fn=compile_fn)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=2
        )
        self.pointwise = nn.Conv2d(in_channels // 2, out_channels, 1, stride, padding)
        self.act_fn = act_fn
        torch.nn.init.zeros_(self.pointwise.weight)
        torch.nn.init.zeros_(self.pointwise.bias)

    def forward(self, x, t=None):
        skip = x
        if self.timestep_embed_size:
            x = self.rms_norm(x, t)
        else:
            x = self.rms_norm(x)
        x = self.conv(x)
        if self.compile_fn:
            x = torch.compile(pointwise_gating(x, self.act_fn))
        else:
            x = pointwise_gating(x, self.act_fn)
        # fused using torch compile
        # lin, gate = rearrange(x, "n (g c) h w -> g n c h w", g=2)
        # x = lin * self.act_fn(gate)
        x = self.pointwise(x)
        return x + skip


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DownBlock, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(scale_factor)
        self.pointwise_conv = nn.Conv2d(
            in_channels * (scale_factor**2), out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = self.pointwise_conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.pointwise_conv = nn.Conv2d(
            in_channels // (scale_factor**2),
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.pointwise_conv(x)
        return x


class ConditionalCrossAttentionBlock(nn.Module):
    def __init__(self):
        super(ConditionalCrossAttentionBlock, self).__init__()
        raise NotImplementedError


class Generator(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=1024,
        layer_count=12,
        pixel_suffle=4,
        timestep_embed_size=None,
        act_fn=torch.sin,
        clamp_output=True,
    ):
        super(Generator, self).__init__()
        self.clamp_output = clamp_output
        self.in_conv = nn.Conv2d(
            in_channels,
            hidden_dim // (pixel_suffle**2),
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.downsample = DownBlock(hidden_dim // (pixel_suffle**2), hidden_dim)
        self.res_blocks = nn.ModuleList()

        for _ in range(layer_count):
            resnet = SimpleResNetBlock(
                hidden_dim, hidden_dim, 3, 1, "same", timestep_embed_size, act_fn
            )
            self.res_block.append(resnet)

        self.upsample = UpBlock(hidden_dim, hidden_dim // (pixel_suffle**2))

        self.out_norm = RMSNorm(hidden_dim)
        self.out_conv = nn.Conv2d(
            hidden_dim // (pixel_suffle**2),
            out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, x, t=None, checkpoint=True):
        x = self.in_conv(x)
        x = self.downsample(x)

        for resnet in self.res_blocks:
            if checkpoint:
                x = ckpt.checkpoint(resnet, x, t)
            else:
                x = resnet(x, t)

        x = self.upsample(x)
        x = self.out_norm(x)
        x = self.out_conv(x)

        if self.clamp_output:
            return F.tanh(x)
        else:
            return x


class UnconditionalGenerator(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=1024,
        layer_count=12,
        pixel_suffle=4,
        timestep_embed_size=None,
        act_fn=torch.sin,
        clamp_output=True,
    ):
        super(UnconditionalGenerator, self).__init__()
        self.clamp_output = clamp_output
        self.in_conv = nn.Conv2d(
            in_channels,
            hidden_dim // (pixel_suffle**2),
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.downsample = DownBlock(hidden_dim // (pixel_suffle**2), hidden_dim)
        self.res_blocks = nn.ModuleList()

        for _ in range(layer_count):
            resnet = SimpleResNetBlock(
                hidden_dim, hidden_dim, 3, 1, "same", timestep_embed_size, act_fn
            )
            self.res_block.append(resnet)

        self.upsample = UpBlock(hidden_dim, hidden_dim // (pixel_suffle**2))

        self.out_norm = RMSNorm(hidden_dim)
        self.out_conv = nn.Conv2d(
            hidden_dim // (pixel_suffle**2),
            out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, x, t=None, checkpoint=True, ckpt_materialize_every=5):
        x = self.in_conv(x)
        x = self.downsample(x)

        for i, resnet in enumerate(self.res_blocks):
            if checkpoint and i + 1 % ckpt_materialize_every != 0:
                x = ckpt.checkpoint(resnet, x, t)
            else:
                x = resnet(x, t)

        x = self.upsample(x)
        x = self.out_norm(x)
        x = self.out_conv(x)

        if self.clamp_output:
            return F.tanh(x)
        else:
            return x


class UnconditionalCritic(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=1,
        hidden_dim=1024,
        layer_count=12,
        pixel_suffle=4,
        timestep_embed_size=None,
        act_fn=torch.sin,
        clamp_output=True,
    ):
        super(UnconditionalCritic, self).__init__()
        self.clamp_output = clamp_output
        self.in_conv = nn.Conv2d(
            in_channels,
            hidden_dim // (pixel_suffle**2),
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.downsample = DownBlock(hidden_dim // (pixel_suffle**2), hidden_dim)
        self.res_blocks = nn.ModuleList()

        for _ in range(layer_count):
            resnet = SimpleResNetBlock(
                hidden_dim, hidden_dim, 3, 1, "same", timestep_embed_size, act_fn
            )
            self.res_block.append(resnet)

        self.upsample = UpBlock(hidden_dim, hidden_dim // (pixel_suffle**2))

        self.out_norm = RMSNorm(hidden_dim)
        self.out_conv = nn.Conv2d(
            hidden_dim // (pixel_suffle**2),
            out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, x, t=None, checkpoint=True, ckpt_materialize_every=5):
        x = self.in_conv(x)
        x = self.downsample(x)

        for i, resnet in enumerate(self.res_blocks):
            if checkpoint and i + 1 % ckpt_materialize_every != 0:
                x = ckpt.checkpoint(resnet, x, t)
            else:
                x = resnet(x, t)

        x = self.upsample(x)
        x = self.out_norm(x)
        x = self.out_conv(x)

        return x
