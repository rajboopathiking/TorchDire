import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import cv2

# 1. Swin Transformer Encoder (pretrained)
class SwinEncoder(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        self.model = create_model('swin_base_patch4_window7_224', pretrained=pretrained, features_only=True)

    def forward(self, x):
        return self.model(x)  # Returns list of features from different layers
    

# 2. ControlNet-like conditioning U-Net decoder
class ControlNetUNet(nn.Module):
    def __init__(self, encoder_channels, hint_channels=1, out_channels=1):
        super().__init__()
        self.condition_conv = nn.Conv2d(hint_channels, encoder_channels[0], 1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[-1], 512, 2, stride=2),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, stride=2),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),   # 56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),    # 112x112 -> 224x224
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 1)              # 224x224 -> 256x256 with interpolation
        )

    def forward(self, features, hint):
        hint_embed = self.condition_conv(hint)


        target_height, target_width = features[0].shape[-2:]
        hint_embed_resized = F.interpolate(hint_embed, size=(target_height, target_width), mode='bilinear', align_corners=False)


        features[0] = features[0] + hint_embed_resized  # Fuse hint with encoder features
        x = features[-1]


        # Transpose if in NHWC format
        if x.shape[1] != 1024:
            x = x.permute(0, 3, 1, 2)  # NHWC to NCHW

        x = self.up(x)
        return x

class SwinControlDiffSeg_Module(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        self.encoder = SwinEncoder(pretrained=pretrained)
        self.decoder = ControlNetUNet(encoder_channels=[56, 256, 512, 1024], out_channels=num_classes)
        self.to_pil = T.ToPILImage()
        self.resize = T.Resize((224, 224))

    def get_canny_hint(self, image_tensor):
        """
        Generate a Canny edge hint from a single image tensor (3xHxW).
        """
        # Convert to PIL and apply resize
        image = self.to_pil(image_tensor.cpu())
        image = self.resize(image)

        # Convert to NumPy for OpenCV
        image_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, threshold1=100, threshold2=200).astype(np.float32) / 255.0
        edges_tensor = torch.from_numpy(edges).unsqueeze(0)  # Shape: [1, H, W]
        return edges_tensor.to(image_tensor.device)

    def forward(self, img):
        """
        img: Tensor of shape [B, 3, H, W]
        """
        # Generate hint (Canny) for each image in batch
        hint_batch = []
        for i in range(img.shape[0]):
            hint = self.get_canny_hint(img[i])
            hint_batch.append(hint)
        hint_tensor = torch.stack(hint_batch, dim=0)  # [B, 1, H, W]

        features = self.encoder(img)
        seg_mask = self.decoder(features, hint_tensor)
        return seg_mask


def SwinControlDiffSeg(num_classes=4, pretrained=False):
    model = SwinControlDiffSeg_Module(num_classes=num_classes,pretrained=pretrained)

    if pretrained:
        try:
            model.load_state_dict(torch.load("Checkpoints/model.pt", map_location="cpu"))
            print("Loaded pretrained weights successfully.")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}. Using randomly initialized weights.")

    return model