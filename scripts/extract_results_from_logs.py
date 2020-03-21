import os
import pandas as pd
from tqdm import tqdm
import pickle
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', '-l', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default='results.pkl')
    parser.add_argument('--eval_files', '-f', nargs='+', type=str,
                        default=['test_accuracy.txt', 'best_val_result.txt'],
                        help='list of evaluation files to read and add to the df')
    parser.add_argument('--eval_names', '-n', nargs='+', type=str,
                        default=['test_accuracy', 'val_accuracy'],
                        help='names to give to results in evaluation files')
    args = parser.parse_args()
    print(args)
    assert len(args.eval_files) == len(args.eval_names)

    logs = os.listdir(args.log_dir)
    df = pd.DataFrame()
    for instance in tqdm(logs):
        if instance == '.gitkeep':
            continue
        try:
            args_file = os.path.join(args.log_dir, instance, 'args.pkl')
            if not os.path.exists(args_file):
                print("===> args.pkl is missing: {}".format(instance))
                continue
            with open(args_file, 'rb') as f:
                result = vars(pickle.load(f))

            file_missing = False
            for file_path, name in zip(args.eval_files, args.eval_names):
                full_file_path = os.path.join(args.log_dir, instance, file_path)
                if not os.path.exists(full_file_path):
                    print("===> {} is missing: {}".format(file_path, instance))
                    file_missing = True
                    break
                with open(full_file_path, 'r') as f:
                    result[name] = float(f.read())

            if not file_missing:
                df = df.append(result, ignore_index=True)

        except Exception as e:
            print(f"\n===> something unexpected went wrong!\n"
                  f"       instance: {instance}\n"
                  f"       error: {e}\n")

    with open(args.output, 'wb') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    main()
