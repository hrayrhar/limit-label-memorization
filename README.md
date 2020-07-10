# Improving generalization by controlling label-noise information in neural network weights

The author implementation of LIMIT method described in the [paper](https://arxiv.org/abs/2002.07933) *"Improving generalization by controlling label-noise information in neural network weights"* by Hrayr Harutyunyan, Kyle Reing, Greg Ver Steeg, and Aram Galstyan.
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
* We use `Pytorch 1.4.0`, but higher versions may work too.
* Additionally, only for extracting data from tensorboard `tensorflow >2.0` is need.

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

The code at the time of ICML 2020 submission code be found [here](https://github.com/hrayrhar/limit-label-memorization/releases/tag/icml).
This is the version of the code at the time of camera-ready submission. This version should be able to reproduce the results in the paper. 

**If you goal is to reproduce the results**, please use this version of the code. The main method of the paper called LIMIT is coded in `methods/predict.py` -- the `PredictGradOutput` class. The soft regularization approach described in the paper corresponds to the class `PenalizeLastLayerFixedForm` of `methods/penalize.py`. The training scripts are generated using the `scripts/generate_commands.py` script.


**If your goal is to use LIMIT**, you can use the newer [code](https://github.com/hrayrhar/limit-label-memorization). It is better commented and easier to use code. The main method there to is the `LIMIT` class of the file `methods/limit.py`.
