from methods.vae import VAE
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

    model = VAE(input_shape=train_loader.dataset[0][0].shape,
                architecture_args=architecture_args,
                device=args.device)

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
