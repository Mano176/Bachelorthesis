from segmentation.unet.unet import UNet
import torch


class InvSegmentater(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(n_channels=3, n_classes=1, bilinear=False)

    def forward(self, input):
        masks_pred = self.unet(input)
        masks_pred = masks_pred.reshape(masks_pred.shape[0], 360, 640)
        return masks_pred