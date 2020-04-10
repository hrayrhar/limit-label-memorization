from modules import utils
from nnlib.nnlib.data_utils.base import load_data_from_arguments
import argparse
import torch
import os
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cuda')

    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dataset', '-D', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'cifar100', 'clothing1m', 'imagenet'])
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

    parser.add_argument('--load_from', type=str, default=None, required=True)
    parser.add_argument('--output_dir', '-o', type=str, default=None)

    args = parser.parse_args()
    print(args)

    # Load data
    _, _, test_loader, _ = load_data_from_arguments(args)

    print(f"Testing the model saved at {args.load_from}")
    model = utils.load(args.load_from, device=args.device)
    ret = utils.apply_on_dataset(model, test_loader.dataset, batch_size=args.batch_size,
                                 output_keys_regexp='pred|label', description='Testing')
    pred = ret['pred']
    labels = ret['label']
    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, 'test_predictions.pkl'), 'wb') as f:
            pickle.dump({'pred': pred, 'labels': labels}, f)

    accuracy = torch.mean((pred.argmax(dim=1) == labels).float())
    print(accuracy)
    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, 'test_accuracy.txt'), 'w') as f:
            f.write("{}\n".format(accuracy))


if __name__ == '__main__':
    main()
