# Sample Checkpoints

This directory contains sample checkpoints in case you want to load a model
without training from scratch.

## celebA

Follow the commands below to load a sample `celebA` model
(assuming your current directory contains `main.py`).

```bash
mkdir -p checkpoint
ln -s checkpoint_demo/demo_celebA_64_64 checkpoint/demo_celebA_64_64
# add any other desired arguments below
python main.py --experiment_name demo --dataset celebA
```

The model was trained on the `celebA` dataset for 10 epochs.
It could be improved by tuning the number of epochs and other hyperparameters.
