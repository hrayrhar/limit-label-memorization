from modules import utils
import modules.data_utils as datasets
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cuda')

    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dataset', '-D', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'cifar100', 'clothing1M'])
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

    args = parser.parse_args()
    print(args)

    # Load data
    train_loader, val_loader, test_loader = datasets.load_data_from_arguments(args)

    print(f"Testing the model saved at {args.load_from}")
    model = utils.load(args.load_from, device=args.device)
    pred = utils.apply_on_dataset(model, test_loader.dataset, batch_size=args.batch_size,
                                  output_keys_regexp='pred', description='Testing')['pred']
    labels = [p[1] for p in test_loader.dataset]
    labels = torch.tensor(labels, dtype=torch.long)
    labels = utils.to_cpu(labels)
    # with open(os.path.join(args.log_dir, 'test_predictions.pkl'), 'wb') as f:
    #     pickle.dump({'pred': pred, 'labels': labels}, f)

    accuracy = torch.mean((pred.argmax(dim=1) == labels).float())
    print(accuracy)
    # with open(os.path.join(args.log_dir, 'test_accuracy.txt'), 'w') as f:
    #     f.write("{}\n".format(accuracy))


if __name__ == '__main__':
    main()
