# Improving generalization by controlling label-noise information in neural network weights
The author implementation of LIMIT method described in the [paper](https://arxiv.org/abs/2002.07933) "Improving generalization by controlling label-noise information in neural network weights" by Hrayr Harutyunyan, Kyle Reing, Greg Ver Steeg, and Aram Galstyan.

**If you goal is to reproduce the results**, please use the version of the code at time of ICML 2020 camera-ready submission.
It can be found in the following [releease](https://github.com/hrayrhar/limit-label-memorization/releases/tag/v0.1). 

**If your goal is to use LIMIT**, you can use the this newer code. It is better commented and easier to use.
The main method of the paper, LIMIT, is coded in the `LIMIT` class of the file `methods/limit.py`.

To cite the paper please use the following BibTeX:
```text
@incollection{harutyunyan2020improving,
 author = {Harutyunyan, Hrayr and Reing, Kyle and Ver Steeg, Greg and Galstyan, Aram},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {5172--5182},
 title = {Improving generalization by controlling label-noise information in neural network weights},
 year = {2020}
}
```

## Requirements:
* Basic data science libraries: `numpy`, `scipy`, `tqdm`, `matplotlib`, `seaborn`, `pandas`, `scikit-learn`.
* We use `Pytorch 1.4.0`, but higher versions should work too.
* Additionally, only for extracting data from tensorboard logs, `tensorflow >= 2.0` is needed.

The exact versions of libraries we used are listed in the `requirements.txt` file.

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

### Structure of the repository
| Directory | Purpose |
|:-----|:----|
| methods | Contains implementations of classifiers used in the paper, including LIMIT.|
| modules | Contains code that is modular and can be shared across different models and experiments.|
| notebooks | Designed for Jupyter notebooks. Contains the notebooks used to generate the plots in the paper. |
| scripts | Contains the scripts for training, testing, collecting results, and generating training commands.|
| configs | Stores training/architecture configurations of our models.|
| logs | Used to store tensorboard logs.|
| data | Used to store data files.|
| plots | Used to store the plots.|
| nnlib | Points to a submodule which contains useful and generic code for training neural networks.| 
