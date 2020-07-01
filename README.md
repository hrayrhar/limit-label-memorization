# Improving generalization by controlling label-noise information in neural network weights

The author implementation of LIMIT method described in the [paper](https://arxiv.org/abs/2002.07933) *"Improving generalization by controlling label-noise information in neural network weights"* by Hrayr Harutyunyan, Kyle Reing, Greg Ver Steeg, and Aram Galstyan.
To cite the paper please use the following BibTeX:
```text
@article{harutyunyan2020improving,
  title={Improving Generalization by Controlling Label-Noise Information in Neural Network Weights},
  author={Harutyunyan, Hrayr and Reing, Kyle and Steeg, Greg Ver and Galstyan, Aram},
  journal={arXiv preprint arXiv:2002.07933},
  year={2020}
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
The current version of the code should also produce the results of the paper.
An updated and better commented code is available in [branch](https://github.com/hrayrhar/limit-label-memorization/tree/nnlib) named `nnlib`.
Please use this code if you need to use LIMIT for new experiments, rather than replicating the paper results.

The main method of the paper called LIMIT is coded in `methods/predict.py` -- the `PredictGradOutput` class.
The soft regularization approach described in the paper corresponds to the class `PenalizeLastLayerFixedForm` of `methods/penalize.py`.
The training scripts are generated using the `scripts/generate_commands.py` script.