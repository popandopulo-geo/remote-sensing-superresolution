# Application of Deep Learning Methods for Enhancing Resolution of Earth Remote Sensing Images

This repository contains the code and models developed as part of my thesis on applying deep learning methods to enhance the resolution of Earth remote sensing images. This work was completed as part of my degree at the Faculty of Computational Mathematics and Cybernetics, Lomonosov Moscow State University.

## Project Overview

The primary goal of this project was to explore and improve neural network-based approaches for enhancing the resolution of Earth remote sensing images, with a focus on applying these methods to satellite imagery. The project involved the implementation and evaluation of several advanced models, including both existing techniques and proposed improvements.

## Implemented Methods

### SRGAN (Super-Resolution Generative Adversarial Network)
A neural network model designed for super-resolution tasks, particularly focused on generating high-resolution images from lower-resolution inputs.

### RCAN (Residual Channel Attention Network)
A deep learning model that utilizes residual learning and channel attention mechanisms to enhance image resolution effectively.

### RCAGAN (Residual Channel Attention Generative Adversarial Network)
A novel model combining the RCAN architecture with the adversarial training approach of SRGAN to improve the quality of high-resolution images.

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
