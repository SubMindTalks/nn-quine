# nn-quine
Implementation of neural network quine (https://arxiv.org/abs/1803.05859).
The idea of a neural network quine is that it is a neural network that reproduces its own weights.
Given a 1-hot vector with a 1 in position _i_, the output of the neural network quine is an approximation of the _i_th parameter of the neural network.
In effect, the neural network reproduces itself.
The original paper didn't publish code, so this is my attempt to reproduce their work.

## Vanilla Quine
The first achievement of this paper was making a neural network quine that just reproduces its own weights.
This is replicated in the file `mains/vanilla_quine.py`.
From the top directory run `python mains/vanilla_quine.py configs/vanilla_quine/00X.json --eval` to reproduce the results (X is a numbered config file in the directory `configs/vanilla_quine/`).
The `--eval` flag is optional; using it will print out some sample parameters after training to show examples of the neural network reproducing its own weights.
To view the training process, use [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) by running `tensorboard --logdir=logs` and navigating to the correct localhost address. Note that this involves having tensorboard installed.

Included are config files for running several different quines of several different sizes. Descriptions are in the json files. Because they are randomly initialized, the exact results are unknown, but several different runs were tried and the quines produced each time looked good.
