# LFDT-Fusion

The code of "LFDT-Fusion: A Latent Feature-guided Diffusion Transformer Model for General Image Fusion"

## Update
- [2024/6] Release model code for LFDT-Fusion.
- 
## To be released
- [ ] Pre-trained model
- [ ] Testing code
- [ ] Training code

## Overview

### Abstract
 For image fusion tasks, it is inefficient for the diffusion model to iterate multiple times on the original resolution image for feature mapping. To address this issue, this paper proposes an efficient latent feature-guided diffusion model for general image fusion. The model consists of a pixel space autoencoder and a compact Transformer-based diffusion network. Specifically, the pixel space autoencoder is a novel UNet-based latent diffusion strategy that compresses inputs into a low-resolution latent space through downsampling. Simultaneously, skip connections transfer multi-scale intermediate features from the encoder to the decoder for decoding, preserving the high-resolution information of the original input. Compared to the existing VAE-GAN-based latent diffusion strategy, the proposed UNet-based strategy is significantly more stable and generates highly detailed images without depending on adversarial optimization. The Transformer-based diffusion network consists of a denoising network and a fusion head. The former captures long-range diffusion dependencies and learns hierarchical diffusion representations, while the latter facilitates diffusion feature interactions to comprehend complex cross-domain information. Moreover, improvements to the diffusion model in noise level, denoising steps, and sampler selection have yielded superior fusion performance across six image fusion tasks. The proposed method illustrates qualitative and quantitative advantages, as evidenced by experimental results in both public datasets and industrial environments.

## Citation
```
@article{,
    author    = {Bo Yang, Zhaohui Jiang, Dong Pan, Haoyang Yu, Gui Gui, Weihua Gui},
    title     = {LFDT-Fusion: A Latent Feature-guided Diffusion Transformer Model for General Image Fusion},
    booktitle = {-},
    month     = {-},
    year      = {-},
    pages     = {-}
}
```
