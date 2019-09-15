from methods import classifiers
from modules import training
import modules.data as datasets
import modules.visualization as vis
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda')

    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--epochs', '-e', type=int, default=400)
    parser.add_argument('--save_iter', '-s', type=int, default=10)
    parser.add_argument('--vis_iter', '-v', type=int, default=2)
    parser.add_argument('--log_dir', '-l', type=str, default=None)

    parser.add_argument('--dataset', '-D', type=str, default='mnist',
                        choices=['mnist', 'cifar10'])
    parser.add_argument('--num_train_examples', type=int, default=None)
    parser.add_argument('--noise_level', '-n', type=float, default=0.0)

    parser.add_argument('--pretrained_vae_path', '-r', type=str, default=None)
    parser.add_argument('--tune_pretrained_parts', dest='freeze_pretrained_parts', action='store_false')
    parser.set_defaults(freeze_pretrained_parts=True)

    parser.add_argument('--loss_function', type=str, default='ce',
                        choices=['ce', 'mse', 'mad'])

    parser.add_argument('--model_class', '-m', type=str, default='StandardClassifier',
                        choices=['StandardClassifier', 'PenalizeLastLayerFixedForm',
                                 'PenalizeLastLayerGeneralForm', 'PredictGradOutputFixedForm',
                                 'PredictGradOutputGeneralForm', 'PredictGradOutputMetaLearning',
                                 'PredictGradOutputGeneralFormUseLabel'])
    parser.add_argument('--grad_weight_decay', '-L', type=float, default=0.0)
    parser.add_argument('--last_layer_l2', type=float, default=0.0)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--nsteps', type=int, default=1)
    args = parser.parse_args()
    print(args)

    # Load data
    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader = datasets.load_mnist_loaders(
            batch_size=args.batch_size, noise_level=args.noise_level,
            num_train_examples=args.num_train_examples)
    if args.dataset == 'cifar10':
        train_loader, val_loader, test_loader = datasets.load_cifar10_loaders(
            batch_size=args.batch_size, noise_level=args.noise_level,
            num_train_examples=args.num_train_examples)

    example_shape = train_loader.dataset[0][0].shape
    print("Dataset is loaded:\n\ttrain_samples: {}\n\tval_samples: {}\n\t"
          "test_samples: {}\n\tsample_shape: {}".format(
        len(train_loader.dataset), len(val_loader.dataset),
        len(test_loader.dataset), example_shape))

    # Options
    optimization_args = {
        'optimizer': {
            'name': 'adam',
            'lr': 1e-3
        }
    }

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

    model_class = getattr(classifiers, args.model_class)

    model = model_class(input_shape=train_loader.dataset[0][0].shape,
                        architecture_args=architecture_args,
                        pretrained_vae_path=args.pretrained_vae_path,
                        device=args.device,
                        freeze_pretrained_parts=args.freeze_pretrained_parts,
                        grad_weight_decay=args.grad_weight_decay,
                        last_layer_l2=args.last_layer_l2,
                        lamb=args.lamb,
                        nsteps=args.nsteps,
                        loss_function=args.loss_function)

    training.train(model=model,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   epochs=args.epochs,
                   save_iter=args.save_iter,
                   vis_iter=args.vis_iter,
                   optimization_args=optimization_args,
                   log_dir=args.log_dir)

    # do final visualizations
    if hasattr(model, 'visualize'):
        visualizations = model.visualize(train_loader, val_loader)

        for name, fig in visualizations.items():
            vis.savefig(fig, os.path.join(args.log_dir, name, 'final.png'))


if __name__ == '__main__':
    main()
