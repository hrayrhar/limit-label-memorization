import os
import sys
import random


def merge_commands(commands, gpu_cnt=10, max_job_cnt=10000, shuffle=True, put_device_id=False):
    sys.stderr.write(f"Created {len(commands)} commands")
    if len(commands) == 0:
        return
    if shuffle:
        random.shuffle(commands)
    merge_cnt = (len(commands) + gpu_cnt - 1) // gpu_cnt
    merge_cnt = min(merge_cnt, max_job_cnt)
    current_device_idx = 0
    for idx in range(0, len(commands), merge_cnt):
        end = min(len(commands), idx + merge_cnt)
        concatenated_commands = "; ".join(commands[idx:end])
        if put_device_id:
            concatenated_commands = concatenated_commands.replace('cuda', f'cuda:{current_device_idx}')
        print(concatenated_commands)
        current_device_idx += 1
        current_device_idx %= gpu_cnt


def check_exists(logdir):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.exists(os.path.join(root_dir, '../', logdir, 'final_test_accuracy.txt'))


def process_command(command):
    arr = command.split(' ')
    logdir = arr[arr.index('-l') + 1]
    if check_exists(logdir):
        sys.stderr.write(f"Skipping {logdir}\n")
        return []
    else:
        return [command]


########################################################################################################################
######################                            CIFAR10-error                                   ######################
########################################################################################################################
ns = [0.2]
ks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
seeds = range(42, 43)
device = 'cuda'
n_epochs = 4000
save_iter = 400
vis_iter = 400
label_noise_type = "error"
dataset = "cifar10"
arch_config = 'configs/double-descent-cifar10-resnet18.json'


""" Standard Classsifier """
method = "StandardClassifier"

commands = []
for n in ns:
    for k in ks:
        for seed in seeds:
            command = f"python -um scripts.train_classifier_double_descent -c {arch_config} -d {device} -e {n_epochs} " \
                f"-s {save_iter} -v {vis_iter} -D {dataset} -n {n} -A --label_noise_type {label_noise_type} -m {method} " \
                f"--seed {seed} -k {k} " \
                f"-l double_descent_logs/{dataset}-{label_noise_type}-noise{n}-augment-{method}-k{k}-seed{seed}"
            commands += process_command(command)

merge_commands(commands, gpu_cnt=10, max_job_cnt=1)


""" PredictGradOutput """
# method = "PredictGradOutput"
# stopping_param = 200
# Ls = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
# q_dists = ['Gaussian', 'Laplace']
#
# # commands = []
# for n in ns:
#     for L in Ls:
#         for q_dist in q_dists:
#             for seed in seeds:
#                 command = f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -A -n {n} --label_noise_type {label_noise_type} -m {method} --q_dist {q_dist} -L {L} --seed {seed} -l logs/{dataset}-{label_noise_type}-noise{n}-augment-{method}-{q_dist}-L{L}-seed{seed}"
#                 commands += process_command(command)
#
# merge_commands(commands, gpu_cnt=10, max_job_cnt=2)


""" PredictGradOutput [loaded] """
# method = "PredictGradOutput"
# stopping_param = 200
# Ls = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
# q_dists = ['Gaussian', 'Laplace']
#
# commands = []
# for n in ns:
#     for L in Ls:
#         for q_dist in q_dists:
#             for seed in seeds:
#                 load_from = f"logs/{dataset}-{label_noise_type}-noise{n}-augment-StandardClassifier-seed{seed}/checkpoints/best_val.mdl"
#                 command = f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -A -n {n} --label_noise_type {label_noise_type} -m {method} --q_dist {q_dist} -L {L} --load_from {load_from} --seed {seed} -l logs/{dataset}-{label_noise_type}-noise{n}-augment-{method}-{q_dist}-L{L}-loaded-seed{seed}"
#                 commands += process_command(command)
#
# merge_commands(commands, gpu_cnt=10, max_job_cnt=2)
