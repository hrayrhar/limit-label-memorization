# Improving generalization by controlling label-noise information in neural network weights

## Structure of the repository
| Directory | Purpose |
|:-----|:----|
| methods | Contains methods, such as classifiers, VAEs, etc. |
| modules | Contains genertic modules of code. |
| notebooks | Designed for storing Jupyter notebooks. |
| scripts | Designed to store scripts -- e.g., *.py with main() or *.sh executables. |
| configs | This stores training/architecture json configuration files. |
| logs | Used to store tensorboard logs. |
| data | Used to store data files. |

## Requirements:
* Basic data science libs: `numpy`, `scipy`, `tqdm`, `matplotlib`, `seaborn`, `pandas`, `scikit-learn`.
* We use `Pytorch 1.2.0`, but higher versions may work too.
* Additionally, only for extracting data from tensorboard files one might need tensorflow.

## Using the code
The whole code is writen as a package. All scripts should be initiated from the root directory.
An example command would be:

```bash
python -um scripts.train_classifier -d cuda -c configs/4layer-cnn-mnist.json --log_dir logs/mnist
```

To monitor the training we run tensorboard:
```bash
tensorboard --logdir=/path/to/the/log/directory
```
