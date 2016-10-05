# dcgan-tfslim
Deep Convolutional Generative Adversarial Networks (DCGAN) implemented in TensorFlow-Slim

Currently under development. Working version borrows code from
https://github.com/carpedm20/DCGAN-tensorflow

# Dependencies
TensorFlow is required. Additionally, install the following dependencies:

```bash
pip install scipy
```

# Setup
Download `celebA` dataset and put images in `data/celebA/`
(create directory structure if needed).

# Train
```bash
python main.py --experiment_name demo --dataset celebA --train True
```

# TensorBoard

Use TensorBoard to visualize losses and generated images.
