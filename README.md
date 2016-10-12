# dcgan-tfslim
Deep Convolutional Generative Adversarial Networks (DCGAN)
implemented in TensorFlow-Slim

This is a TensorFlow implementation of the following paper:
https://arxiv.org/pdf/1511.06434v2.pdf.
**Some parameters and settings may not be exactly the same from the paper!**
Nonetheless, the code is able to generate images.

# Dependencies
TensorFlow and `tensorflow.contrib.slim` are required, along with their
dependencies (e.g. numpy). The only other additional dependency is PIL.
This can be installed with pip:

```bash
pip install Pillow
```

# Setup: celebA
Download the [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
dataset and put the images in `data/celebA/`
(create the directory structure if needed).

# Setup: your own dataset
Put your images in `data/your_dataset/`. Create the directory structure and
name `your_dataset` with whatever you want. Images should be `*.jpg`.

Optionally, create a file `data/your_dataset.txt` with each line
containing an image file name in `data/your_dataset/*`. (One way
to generate this file is to run the following command in the `data/` directory:
`ls your_dataset/ > your_dataset.txt`.)
This file specifies the order of training. If this file is not found,
then a random order will be used.

# Train
Train on celebA:

```bash
python main.py --experiment_name celebA_demo --dataset celebA --train True
```

Train on your dataset:

```bash
python main.py --experiment_name your_dataset_demo --dataset your_dataset --train True
```

The `--dataset` flag accepts whatever dataset folder you want to use in the `data/` directory.

# TensorBoard

Use TensorBoard to visualize losses and generated images.
