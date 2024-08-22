# Application of Deep Learning Methods for Enhancing Resolution of Earth Remote Sensing Images

This repository contains the code and models developed as part of my thesis on applying deep learning methods to enhance the resolution of Earth remote sensing images. This work was completed as part of my degree at the Faculty of Computational Mathematics and Cybernetics, Lomonosov Moscow State University.

## Project Overview

The primary goal of this project was to explore and improve neural network-based approaches for enhancing the resolution of Earth remote sensing images, with a focus on applying these methods to satellite imagery. The project involved the implementation and evaluation of several advanced models, including both existing techniques and proposed improvements.

## Implemented Methods


**Super-Resolution Generative Adversarial Network (SRGAN):** A neural network approach for image super-resolution that utilizes a generative adversarial network (GAN) to produce high-resolution images from low-resolution inputs.

**Residual Channel Attention Network (RCAN):** A deep learning model that combines residual learning with channel attention mechanisms to effectively enhance image resolution by focusing on important features.

### Modifications and Improvements

**Residual Channel Attention Generative Adversarial Network (RCAGAN):** A novel model that integrates the RCAN architecture with adversarial training from SRGAN, designed to improve image resolution further. This hybrid model leverages the strengths of both approaches and represents my own modification to enhance the super-resolution task.


## Data Used

- **xView**: A dataset consisting of high-resolution satellite images used for training and evaluating the super-resolution models.
- **Sentinel-2**: Satellite imagery used for testing the application of the developed models on different data sources.

## Key Improvements

- **Hybrid Model Architecture**: Integration of RCAN with the adversarial training from SRGAN to form the RCAGAN model, leveraging the strengths of both approaches.
- **Data Augmentation**: Implemented various data augmentation techniques such as random cropping, rotations, and reflections to improve model generalization.

## Results

The modified models demonstrated superior performance in enhancing image resolution, with improvements observed in PSNR and SSIM metrics across multiple datasets. The enhanced resolution models also showed the potential to be applied to images from different satellites, such as Sentinel-2, with promising results.

## Thesis Text and Results

The full thesis text, including detailed descriptions of the methods, experiments, and results, can be found in the file `text.pdf`. The document is written in Russian.

## Acknowledgements

This project was completed under the supervision of Dr. V.V. Glazkova at the Faculty of Computational Mathematics and Cybernetics, Lomonosov Moscow State University. I would like to thank my advisor and the research group for their support and guidance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
