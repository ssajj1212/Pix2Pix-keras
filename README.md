# Pix2Pix-keras
Modified Pix2Pix keras implementation adding perceptual loss.

## Geting Started

## Prerequisites

## Generative Adversarial Networks
Discriminator discriminates whether image is fake or real; generator generates synthetic images
GAN framework learns the loss function for the real image distribution to make the generator images more realistic but still not deterministic mapping
Loss function includes GAN loss, L1 loss and perceptual loss

<img src="images/Pix2pixTrainProcess.png" width="900px"/>
## Networks Architectures

### Generator with skips
U-Net with skip connection links the layer i in encoder to the layer (n-i) in decoder, to preserve some low-level representation between input and output domain
<img src="images/Unet.png" width="900px"/>

### PatchGAN
Markovian discriminator, classifies NxN patches and average the classification result for whole image; smaller discriminator and faster training and inference
<img src="images/PatchGAN.png" width="900px"/>



## Examples

### Cityscapes vs BDD100K
<img src="images/CityscapeBDD.png" width="900px"/>


### Cityscapes

### BDD100K

### Simulation scenarios from Carla simulator
<img src="images/Sim.png" width="900px"/>


### Manipulated scenarios
<img src="images/Manipulated.png" width="900px"/>


## References
