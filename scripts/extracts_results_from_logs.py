import os
import pandas as pd
from tqdm import tqdm
import pickle
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', '-l', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default='results.pkl')
    args = parser.parse_args()
    print(args)

    logs = os.listdir(args.log_dir)
    df = pd.DataFrame()
    for instance in tqdm(logs):
        if instance == '.gitkeep':
            continue

        with open(os.path.join(args.log_dir, instance, 'args.pkl'), 'rb') as f:
            result = vars(pickle.load(f))

        with open(os.path.join(args.log_dir, instance, 'test_accuracy.txt'), 'r') as f:
            test_acc = float(f.read())
            result['test_accuracy'] = test_acc

        val_result_file_path = os.path.join(args.log_dir, instance, 'best_val_result.txt')
        if os.path.exists(val_result_file_path):
            with open(val_result_file_path, 'r') as f:
                val_acc = float(f.read())
        else:
            val_acc = -1.0
        result['val_accuracy'] = val_acc

        df = df.append(result, ignore_index=True)

    with open(args.output, 'wb') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    main()
