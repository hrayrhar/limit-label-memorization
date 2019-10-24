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
    parser.add_argument('--transform_function', type=str, default=None,
                        choices=[None, 'remove_random_chunks'])
    parser.add_argument('--transform_validation', dest='transform_validation', action='store_true')
    parser.add_argument('--no-transform_validation', dest='transform_validation', action='store_false')
    parser.set_defaults(transform_validation=True)
    parser.add_argument('--remove_prob', type=float, default=0.5)

    parser.add_argument('--model_class', '-m', type=str, default='StandardClassifier',
                        choices=['StandardClassifier', 'PenalizeLastLayerFixedForm',
                                 'PenalizeLastLayerGeneralForm', 'PredictGradOutputFixedForm',
                                 'PredictGradOutputGeneralForm', 'PredictGradOutputMetaLearning',
                                 'PredictGradOutputGeneralFormUseLabel'])
    parser.add_argument('--loss_function', type=str, default='ce',
                        choices=['ce', 'mse', 'mad'])
    parser.add_argument('--grad_weight_decay', '-L', type=float, default=0.0)
    parser.add_argument('--grad_l1_penalty', '-S', type=float, default=0.0)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--nsteps', type=int, default=1)
    parser.add_argument('--pretrained_arg', '-r', type=str, default=None)
    args = parser.parse_args()
    print(args)

    # Load data
    train_loader, val_loader, test_loader = datasets.load_data_from_arguments(args)

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
                        pretrained_arg=args.pretrained_arg,
                        device=args.device,
                        grad_weight_decay=args.grad_weight_decay,
                        grad_l1_penalty=args.grad_l1_penalty,
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
                   log_dir=args.log_dir,
                   args_to_log=args)

    # do final visualizations
    if hasattr(model, 'visualize'):
        visualizations = model.visualize(train_loader, val_loader)

        for name, fig in visualizations.items():
            vis.savefig(fig, os.path.join(args.log_dir, name, 'final.png'))


if __name__ == '__main__':
    main()
