import methods
from modules import training, utils
import modules.data_utils as datasets
import argparse
import pickle
import torch
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda')

    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--epochs', '-e', type=int, default=400)
    parser.add_argument('--stopping_param', type=int, default=50)
    parser.add_argument('--save_iter', '-s', type=int, default=10)
    parser.add_argument('--vis_iter', '-v', type=int, default=10)
    parser.add_argument('--log_dir', '-l', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dataset', '-D', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'clothing1M'])
    parser.add_argument('--data_augmentation', '-A', action='store_true', dest='data_augmentation')
    parser.set_defaults(data_augmentation=False)
    parser.add_argument('--num_train_examples', type=int, default=None)
    parser.add_argument('--label_noise_level', '-n', type=float, default=0.0)
    parser.add_argument('--label_noise_type', type=str, default='flip',
                        choices=['flip', 'error', 'cifar10_custom'])
    parser.add_argument('--transform_function', type=str, default=None,
                        choices=[None, 'remove_random_chunks'])
    parser.add_argument('--clean_validation', dest='clean_validation', action='store_true')
    parser.set_defaults(clean_validation=False)
    parser.add_argument('--remove_prob', type=float, default=0.5)

    parser.add_argument('--model_class', '-m', type=str, default='StandardClassifier')
    parser.add_argument('--loss_function', type=str, default='ce',
                        choices=['ce', 'mse', 'mae', 'gce', 'dmi', 'fw', 'none'])
    parser.add_argument('--loss_function_param', type=float, default=1.0)
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--grad_weight_decay', '-L', type=float, default=0.0)
    parser.add_argument('--grad_l1_penalty', '-S', type=float, default=0.0)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--pretrained_arg', '-r', type=str, default=None)
    parser.add_argument('--sample_from_q', action='store_true', dest='sample_from_q')
    parser.set_defaults(sample_from_q=False)
    parser.add_argument('--q_dist', type=str, default='Gaussian', choices=['Gaussian', 'Laplace', 'dot'])
    parser.add_argument('--no-detach', dest='detach', action='store_false')
    parser.set_defaults(detach=True)

    parser.add_argument('--add_noise', action='store_true', dest='add_noise',
                        help='add noise to the gradients of a standard classifier.')
    parser.set_defaults(add_noise=False)
    parser.add_argument('--noise_type', type=str, default='Gaussian', choices=['Gaussian', 'Laplace'])
    parser.add_argument('--noise_std', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    print(args)

    # Load data
    train_loader, val_loader, test_loader = datasets.load_data_from_arguments(args)

    # Options
    optimization_args = {
        'optimizer': {
            'name': 'adam',
            'lr': args.lr
        }
    }

    # optimization_args = {
    #     'optimizer': {
    #         'name': 'sgd',
    #         'lr': 1e-3,
    #     },
    #     'scheduler': {
    #         'step_size': 15,
    #         'gamma': 1.25
    #     }
    # }

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

    model_class = getattr(methods, args.model_class)

    model = model_class(input_shape=train_loader.dataset[0][0].shape,
                        architecture_args=architecture_args,
                        pretrained_arg=args.pretrained_arg,
                        device=args.device,
                        grad_weight_decay=args.grad_weight_decay,
                        grad_l1_penalty=args.grad_l1_penalty,
                        lamb=args.lamb,
                        sample_from_q=args.sample_from_q,
                        q_dist=args.q_dist,
                        load_from=args.load_from,
                        loss_function=args.loss_function,
                        loss_function_param=args.loss_function_param,
                        add_noise=args.add_noise,
                        noise_type=args.noise_type,
                        noise_std=args.noise_std,
                        detach=args.detach)

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

    # if training finishes successfully, compute the test score
    print("Testing the best validation model...")
    model = utils.load(os.path.join(args.log_dir, 'checkpoints', 'best_val.mdl'),
                       device=args.device)
    pred = utils.apply_on_dataset(model, test_loader.dataset, batch_size=args.batch_size,
                                  output_keys_regexp='pred', description='Testing')['pred']
    labels = [p[1] for p in test_loader.dataset]
    labels = torch.tensor(labels, dtype=torch.long)
    labels = utils.to_cpu(labels)
    with open(os.path.join(args.log_dir, 'test_predictions.pkl'), 'wb') as f:
        pickle.dump({'pred': pred, 'labels': labels}, f)

    accuracy = torch.mean((pred.argmax(dim=1) == labels).float())
    with open(os.path.join(args.log_dir, 'test_accuracy.txt'), 'w') as f:
        f.write("{}\n".format(accuracy))


if __name__ == '__main__':
    main()
