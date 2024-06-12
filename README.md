# DeepLabV3 Rock Segmentation

This project implements a rock segmentation model using DeepLabV3 with a ResNet-50 backbone. The model is trained to detect and segment rocks in images and videos.

## Table of Contents
- [Project Description](#project-description)
- [Setup Instructions](#setup-instructions)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [References](#references)

## Project Description

DeepLabV3 is a state-of-the-art semantic segmentation model that utilizes atrous convolution to increase the field-of-view of filters without increasing the number of parameters. This project trains a DeepLabV3 model to segment rocks from images and videos.

## Setup Instructions

### Prerequisites

- Python 3.6+
- PyTorch
- Torchvision
- OpenCV
- Pillow
- tqdm
- check the deeplabv3 env folder for more details

## Model Architecture

DeepLabV3 with ResNet-50 backbone is used for the segmentation task. Below is a detailed overview of the model architecture:

| Layer Type            | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| Input Layer           | Accepts RGB images of shape (3, H, W)                                                         |
| ResNet-50 Backbone    | A deep residual network that extracts features from the input image. It consists of:          |
|                       | - **Conv1**: Initial convolution layer with BatchNorm and ReLU activation.                    |
|                       | - **Res2, Res3, Res4, Res5**: Four stages of residual blocks with shortcut connections.        |
| Atrous Convolution    | Applies dilated convolution to capture multi-scale contextual information without losing resolution. |
|                       | - **Atrous Conv Layers**: Convolutional layers with different dilation rates.                 |
| ASPP Module           | Atrous Spatial Pyramid Pooling module for multi-scale context aggregation.                    |
|                       | - **ASPP Conv Layers**: Multiple atrous convolutions with different rates and global pooling.  |
|                       | - **Concat & Conv**: Concatenates all the atrous convolution results and applies a 1x1 convolution. |
| Decoder               | Upsamples the low-resolution feature maps to the original image size.                         |
|                       | - **Upsampling**: Bilinear upsampling layers.                                                 |
|                       | - **Skip Connections**: Connects lower-level features from the backbone to the decoder.       |
| Output Layer          | Produces the segmentation mask with the same spatial dimensions as the input image.           |

### Detailed Explanation of Layers:

- **ResNet-50 Backbone**:
  - **Conv1**: This layer performs a standard convolution operation followed by batch normalization and ReLU activation.
  - **Residual Blocks (Res2, Res3, Res4, Res5)**: These blocks consist of several layers of convolutions, batch normalization, and ReLU activations with shortcut connections that add the input of the block to the output. This helps in training deep networks by mitigating the vanishing gradient problem.

- **Atrous Convolution**: 
  - Uses dilated convolutions which allow the network to have a larger receptive field without increasing the number of parameters. This is crucial for capturing multi-scale context in the image.

- **ASPP Module**:
  - Combines multiple atrous convolutions with different dilation rates and a global pooling layer. This helps in capturing information at various scales, which is essential for precise segmentation.

- **Decoder**:
  - Upsamples the feature maps to the original input image size using bilinear upsampling. It also includes skip connections from earlier layers in the network to help retain fine spatial information.

## When to Use DeepLabV3

### Suitable Use Cases:
- **High-Resolution Images**: DeepLabV3 is effective for segmenting high-resolution images, where capturing detailed context is crucial.
- **Multi-Scale Context**: The model excels in scenarios where capturing information at multiple scales is important, thanks to its atrous convolutions and ASPP module.
- **Object Segmentation**: Ideal for tasks involving object segmentation in images, such as road segmentation, medical imaging


## References

- [DeepLabV3 Paper](https://arxiv.org/abs/1706.05587)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [OpenCV Documentation](https://docs.opencv.org/)

