# dcgan-tfslim
Deep Convolutional Generative Adversarial Networks (DCGAN) implemented in TensorFlow-Slim

Currently under development. Working version borrows code from
https://github.com/carpedm20/DCGAN-tensorflow

# Dependencies
TensorFlow and `tensorflow.contrib.slim` is required.
The only other additional dependency is PIL. This can be installed with pip:

```bash
pip install Pillow
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
