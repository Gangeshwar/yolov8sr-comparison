import torch
from ultralytics import YOLO

class YOLOv8SR(YOLO):   
    """
    YOLOv8SR: Customized YOLO model for both object detection with super-resolution.
    
    Wide Activation for Efficient and Accurate Image Super-Resolution (WDSR),

    This class inherits from the Ultralytics YOLO model and extends it by adding
    custom layers specifically designed for super-resolution tasks. It's suitable for
    scenarios where enhanced image resolution is crucial along with object detection,
    such as satellite imagery, medical imaging, or high-quality surveillance.

    Attributes:
        custom_conv1 (torch.nn.Module): First custom convolutional layer for feature extraction.
        batch_norm1 (torch.nn.Module): Batch normalization layer for the first custom convolutional layer.
        relu1 (torch.nn.Module): ReLU activation for non-linear transformation after the first convolutional layer.
        maxpool1 (torch.nn.Module): Max pooling layer to reduce spatial dimensions after the first convolutional layer.
        custom_conv2 (torch.nn.Module): Second custom convolutional layer for deeper feature extraction.
        batch_norm2 (torch.nn.Module): Batch normalization layer for the second custom convolutional layer.
        relu2 (torch.nn.Module): ReLU activation for non-linear transformation after the second convolutional layer.

    Methods:
        forward(x): Defines the forward pass combining the YOLO base model with the custom super-resolution layers.
    """

    def __init__(self, cfg='train.yaml', pretrained_weights = "yolov8s.pt"):
        # Initialize the base YOLO model
        super(YOLO, self).__init__(model=pretrained_weights)
        
        self.yolo = YOLO(pretrained_weights)

        # Additional Conv layers for Super Resolution   
        # Wide Activation for Efficient and Accurate Image Super-Resolution (WDSR),     
    

        self.custom_conv1 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(1024)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.custom_conv2 = torch.nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(2048)
        self.relu2 = torch.nn.ReLU()

        # Upsampling layer to upscale the image tensor
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

           
        #    torch.optim.optimizer.Optimizer
    def forward(self, layer):
        # Base model features
        layer = super(YOLOv8SR, self).forward(layer)

        # Custom layers
        layer = self.custom_conv1(layer)
        layer = self.batch_norm1(layer)
        layer = self.relu1(layer)
        layer = self.maxpool1(layer)

        layer = self.custom_conv2(layer)
        layer = self.batch_norm2(layer)
        layer = self.relu2(layer)

        # Upsampling step to increase the resolution of the image tensor
        layer = self.upsample(layer)
        
        return layer
    

if __name__ == "__main__":
    # Load pre-trained weights (ensure the model architecture matches the weights)
    pretrained_weights = 'yolov8s.pt'
    cfg='train.yaml'
    # Create an instance of the customYOLOv8  model
    yolov8sr_model = YOLOv8SR(cfg=cfg, pretrained_weights = pretrained_weights)

