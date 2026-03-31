import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPad(nn.Module):
    def __init__(self, pad_height, pad_width):
        super().__init__()
        self.pad_height = pad_height
        self.pad_width = pad_width

    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad_height, self.pad_height), mode="reflect")
        x = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        return x


class ConvResBlockSingle(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, p_drop=0.0, gn_groups=1):
        super().__init__()
        pad = (k_size - 1) // 2
        self.pad = CustomPad(pad, pad)
        self.conv1 = nn.Conv2d(in_ch, out_ch, k_size, padding=0)
        self.gn1 = nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch)
        self.act = nn.Mish(inplace=True)
        self.dp = nn.Dropout2d(p_drop) if p_drop else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = self.act(self.gn1(self.conv1(self.pad(x))))
        y = self.dp(y)
        return self.act(y + self.skip(x))


class Unet6R(nn.Module):
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        base_channels=32,
        kernel_size=3,
        p_drop=0.1,
        gn_groups=1,
    ):
        super().__init__()
        k = kernel_size
        c1, c2, c4, c8, c16, c32, c64 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
            base_channels * 32,
            base_channels * 64,
        )

        self.enc1 = ConvResBlockSingle(input_channels, c1, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvResBlockSingle(c1, c2, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvResBlockSingle(c2, c4, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvResBlockSingle(c4, c8, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = ConvResBlockSingle(c8, c16, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.pool5 = nn.MaxPool2d(2)
        self.enc6 = ConvResBlockSingle(c16, c32, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.pool6 = nn.MaxPool2d(2)

        self.bottleneck = ConvResBlockSingle(c32, c64, k_size=k, p_drop=p_drop, gn_groups=gn_groups)

        self.upconv1 = nn.ConvTranspose2d(c64, c32, kernel_size=2, stride=2)
        self.dec1 = ConvResBlockSingle(c32 + c32, c32, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.upconv2 = nn.ConvTranspose2d(c32, c16, kernel_size=2, stride=2)
        self.dec2 = ConvResBlockSingle(c16 + c16, c16, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.upconv3 = nn.ConvTranspose2d(c16, c8, kernel_size=2, stride=2)
        self.dec3 = ConvResBlockSingle(c8 + c8, c8, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.upconv4 = nn.ConvTranspose2d(c8, c4, kernel_size=2, stride=2)
        self.dec4 = ConvResBlockSingle(c4 + c4, c4, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.upconv5 = nn.ConvTranspose2d(c4, c2, kernel_size=2, stride=2)
        self.dec5 = ConvResBlockSingle(c2 + c2, c2, k_size=k, p_drop=p_drop, gn_groups=gn_groups)
        self.upconv6 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec6 = ConvResBlockSingle(c1 + c1, c1, k_size=k, p_drop=p_drop, gn_groups=gn_groups)

        self.final_conv = nn.Conv2d(c1, output_channels, kernel_size=1)

    def forward(self, x):
        original_h, original_w = x.shape[2], x.shape[3]
        pad_h = (64 - original_h % 64) % 64
        pad_w = (64 - original_w % 64) % 64

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                mode="reflect",
            )

        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))
        x6 = self.enc6(self.pool5(x5))
        b = self.bottleneck(self.pool6(x6))

        d1 = self.dec1(torch.cat([self.upconv1(b), x6], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d1), x5], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d2), x4], dim=1))
        d4 = self.dec4(torch.cat([self.upconv4(d3), x3], dim=1))
        d5 = self.dec5(torch.cat([self.upconv5(d4), x2], dim=1))
        d6 = self.dec6(torch.cat([self.upconv6(d5), x1], dim=1))

        output = self.final_conv(d6)
        if pad_h > 0 or pad_w > 0:
            output = output[
                :,
                :,
                pad_h // 2 : original_h + pad_h // 2,
                pad_w // 2 : original_w + pad_w // 2,
            ]
        return output


class SoftmaxHead(nn.Module):
    def __init__(self, in_channels, num_bins):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_bins, kernel_size=1)

    def forward(self, feat):
        logits = self.conv(feat)
        return torch.softmax(logits, dim=1)


class ProbUNet(nn.Module):
    def __init__(self, input_channels, base_channels, kernel_size, p_drop, num_bins, gn_groups=1):
        super().__init__()
        self.backbone = Unet6R(
            input_channels=input_channels,
            output_channels=base_channels,
            base_channels=base_channels,
            kernel_size=kernel_size,
            p_drop=p_drop,
            gn_groups=gn_groups,
        )
        self.head = SoftmaxHead(base_channels, num_bins)

    def forward(self, x):
        return self.head(self.backbone(x))

    def forward_components(self, x):
        return self.forward(x)
