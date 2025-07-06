import torch
from ultralytics import YOLO
from torch import nn

# --------------------------
# Residual Block for EDSR
# --------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, layer):
        residual = self.relu(self.conv1(layer))
        residual = self.conv2(residual)
        return layer + residual

# --------------------------
# EDSR Super-Resolution Module
# --------------------------
class EDSR(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_blocks=5, scale=2):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, layer):
        layer = self.head(layer)
        layer = self.body(layer)
        layer = self.tail(layer)
        return layer

# --------------------------
# InceptionNeXt Block (Custom Replacement for C2F)
# --------------------------
class InceptionNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        self.branch1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        self.branch3 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels)
        self.branch4 = nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3, groups=mid_channels)

        self.conv_out = nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, layer):
        x1 = self.branch1(layer)
        x2 = self.branch2(layer)
        x3 = self.branch3(layer)
        x4 = self.branch4(layer)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv_out(out)
        out = self.norm(out)
        out = self.relu(out)
        return out

# --------------------------
# Custom YOLOv8SR Class
# --------------------------  
class YOLOv8SR(YOLO):   
    """

    YOLOv8SR: YOLOv8 extended with EDSR super-resolution and InceptionNeXt backbone block.
    This class inherits from the Ultralytics YOLO model and extends it by adding
    an EDSR super-resolution module. It's suitable for scenarios where enhanced image
    resolution is crucial along with object detection, such as satellite imagery,
    medical imaging, or high-quality surveillance.

    """

    def __init__(self, cfg='train.yaml', pretrained_weights = "yolov8s.pt"):
        # Initialize the base YOLO model
        super().__init__(pretrained_weights)
        
        # Define EDSR module
        self.edsr = EDSR()

        # Replace one of the C2F blocks in the backbone with InceptionNeXt
        # You can inspect which layer to replace via: print(self.model.model)
        # Replace one backbone layer with InceptionNeXtBlock
        backbone_layers = self.model.model[0]  # Backbone is usually model[0]
        print("Original backbone structure:\n", backbone_layers)

        # Replace 3rd layer (index = 2) in backbone — adjust index if needed
        c2f_layer_index = 2
        if hasattr(backbone_layers[c2f_layer_index], 'conv'):
            in_channels = backbone_layers[c2f_layer_index].conv[0].in_channels
            out_channels = backbone_layers[c2f_layer_index].conv[-1].out_channels

            backbone_layers[c2f_layer_index] = InceptionNeXtBlock(in_channels, out_channels)
            print(f"Replaced backbone layer {c2f_layer_index} with InceptionNeXtBlock")
        else:
            print(f"Warning: Expected 'conv' layer at index {c2f_layer_index}, but not found.")
  
    def forward(self, layer):
        # Apply EDSR to enhance input resolution
        layer = self.edsr(layer)
        
        # Pass through YOLOv8 model
        layer = super().forward(layer)
        return layer
    

if __name__ == "__main__":
    # Load pre-trained weights (ensure the model architecture matches the weights)
    pretrained_weights = 'yolov8s.pt'
    cfg='train.yaml'
    # Create an instance of the customYOLOv8  model
    yolov8sr_model = YOLOv8SR(cfg=cfg, pretrained_weights = pretrained_weights)
    
