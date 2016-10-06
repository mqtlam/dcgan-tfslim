# dcgan-tfslim
Deep Convolutional Generative Adversarial Networks (DCGAN)
implemented in TensorFlow-Slim

This is a TensorFlow implementation of the following paper:
https://arxiv.org/pdf/1511.06434v2.pdf.
Some parameters and settings may not be exactly the same from the paper.

# Dependencies
TensorFlow and `tensorflow.contrib.slim` is required, along with their
dependencies (e.g. numpy). The only other additional dependency is PIL.
This can be installed with pip:

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
