# Reducing overfitting by blocking label noise information

## Structure of the repository
| Directory | Purpose |
|:-----|:----|
| methods | Here we keep the models we develop, like VAE with sparse generator. |
| modules | Here we keep codes that are modular and can be shared across different models and experiments.|
| notebooks | This is designed for Jupyter notebooks. |
| scripts | This one is designed to keep all scripts, everything that has a main() function should live here.|
| configs | This stores training/architecture configurations of our models, so that we reduces hardcoded parts.|
| logs | Used to store tensorboard logs.|
| data | Used to store data files.|


## Installation
The code requires basic data science libraries, such as `numpy`, `scipy`, `tqdm`, `matplotlib`, `scikit-learn`, etc.
For training the models we use `Pytorch`, which can be installed using the following command:
``` bash
conda install pytorch torchvision -c pytorch
```
For monitoring the training `tensorboard` and `tensorboardX` are needed.
```bash
pip install tensorflow  # or tensorflow-gpu, this will install tensorboard
pip install tensorboardX
```

## Using the code
The whole code is writen as a package. All scripts should be initiated from the root directory.
An example command would be:

```bash
python -um scripts.train-classifier -d cuda -c configs/classifier-mnist.json --log_dir logs/mnist;
```

To monitor the training we run tensorboard:
```bash
tensorboard --logdir=/path/to/the/log/directory
```
