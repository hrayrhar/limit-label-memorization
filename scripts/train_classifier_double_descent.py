import methods
from modules import training, utils
from nnlib.nnlib.data_utils.base import load_data_from_arguments
import argparse
import pickle
import torch
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda')

    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--epochs', '-e', type=int, default=4000)
    parser.add_argument('--stopping_param', type=int, default=2**30)
    parser.add_argument('--save_iter', '-s', type=int, default=100)
    parser.add_argument('--vis_iter', '-v', type=int, default=10)
    parser.add_argument('--log_dir', '-l', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dataset', '-D', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'clothing1m', 'imagenet'])
    parser.add_argument('--data_augmentation', '-A', action='store_true', dest='data_augmentation')
    parser.set_defaults(data_augmentation=False)
    parser.add_argument('--num_train_examples', type=int, default=None)
    parser.add_argument('--label_noise_level', '-n', type=float, default=0.0)
    parser.add_argument('--label_noise_type', type=str, default='error',
                        choices=['error', 'cifar10_custom'])
    parser.add_argument('--transform_function', type=str, default=None,
                        choices=[None, 'remove_random_chunks'])
    parser.add_argument('--clean_validation', dest='clean_validation', action='store_true')
    parser.set_defaults(clean_validation=False)
    parser.add_argument('--remove_prob', type=float, default=0.5)

    parser.add_argument('--model_class', '-m', type=str, default='StandardClassifier')
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--grad_weight_decay', '-L', type=float, default=0.0)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--pretrained_arg', '-r', type=str, default=None)
    parser.add_argument('--sample_from_q', action='store_true', dest='sample_from_q')
    parser.set_defaults(sample_from_q=False)
    parser.add_argument('--q_dist', type=str, default='Gaussian', choices=['Gaussian', 'Laplace', 'dot'])
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    parser.add_argument('--k', '-k', type=int, required=True, default=10,
                        help='width parameter of ResNet18-k')
    args = parser.parse_args()
    print(args)

    # Load data
    train_loader, val_loader, test_loader, _ = load_data_from_arguments(args)

    # Options
    optimization_args = {
        'optimizer': {
            'name': 'adam',
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }
    }

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

        # set the width parameter k
        if ('classifier' in architecture_args and
                architecture_args['classifier'].get('net', '') == 'double-descent-cifar10-resnet18'):
            architecture_args['classifier']['k'] = args.k
        if ('q-network' in architecture_args and
                architecture_args['q-network'].get('net', '') == 'double-descent-cifar10-resnet18'):
            architecture_args['q-network']['k'] = args.k

    model_class = getattr(methods, args.model_class)

    model = model_class(input_shape=train_loader.dataset[0][0].shape,
                        architecture_args=architecture_args,
                        pretrained_arg=args.pretrained_arg,
                        device=args.device,
                        grad_weight_decay=args.grad_weight_decay,
                        lamb=args.lamb,
                        sample_from_q=args.sample_from_q,
                        q_dist=args.q_dist,
                        load_from=args.load_from,
                        loss_function='ce')

    training.train(model=model,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   epochs=args.epochs,
                   save_iter=args.save_iter,
                   vis_iter=args.vis_iter,
                   optimization_args=optimization_args,
                   log_dir=args.log_dir,
                   args_to_log=args,
                   stopping_param=args.stopping_param)

    # test the last model and best model
    models_to_test = [
        {
            'name': 'best',
            'file': 'best_val.mdl'
        },
        {
            'name': 'final',
            'file': 'final.mdl'
        }
    ]
    for spec in models_to_test:
        print("Testing the {} model...".format(spec['name']))
        model = utils.load(os.path.join(args.log_dir, 'checkpoints', spec['file']),
                           device=args.device)
        pred = utils.apply_on_dataset(model, test_loader.dataset, batch_size=args.batch_size,
                                      output_keys_regexp='pred', description='Testing')['pred']
        labels = [p[1] for p in test_loader.dataset]
        labels = torch.tensor(labels, dtype=torch.long)
        labels = utils.to_cpu(labels)
        with open(os.path.join(args.log_dir, '{}_test_predictions.pkl'.format(spec['name'])), 'wb') as f:
            pickle.dump({'pred': pred, 'labels': labels}, f)

        accuracy = torch.mean((pred.argmax(dim=1) == labels).float())
        with open(os.path.join(args.log_dir, '{}_test_accuracy.txt'.format(spec['name'])), 'w') as f:
            f.write("{}\n".format(accuracy))


if __name__ == '__main__':
    main()
