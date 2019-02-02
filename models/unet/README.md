# U-Net Implementation

PyTorch implementation of the network found in clinically applicable deep learning paper (De Fauw et al. (2018)). Image taken from supplementary information.

![Network to be implemented](https://raw.githubusercontent.com/j-bernardi/dnn-information/master/docs/supplementary/u-net.png)

Ref: Nature Medicine, volume 24, pages1342–1350 (2018)

## Credits

See here for info on U-Nets:

https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

3D U-Nets:
https://arxiv.org/pdf/1606.06650.pdf

Implementation based on

Based on: https://github.com/shiba24/3d-unet

## Scripts
### unet_model.py
Sets out the model structure as found here  (supplementary figure 14):

https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-018-0107-6/MediaObjects/41591_2018_107_MOESM1_ESM.pdf

### train_segment.py
Train the segmentation network.
