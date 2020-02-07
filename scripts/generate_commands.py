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
    return os.path.exists(os.path.join(root_dir, '../', logdir, 'test_accuracy.txt'))


def process_command(command):
    arr = command.split(' ')
    logdir = arr[arr.index('-l') + 1]
    if check_exists(logdir):
        sys.stderr.write(f"Skipping {logdir}\n")
        return []
    else:
        return [command]


########################################################################################################################
########################                            MNIST-error                                   ######################
########################################################################################################################
ns = [0.0, 0.5, 0.8, 0.89]
seeds = range(42, 47)
n_samples_grid = [1000, 10000, None]
device = 'cuda'
n_epochs = 400
save_iter = 10000
vis_iter = 50
stopping_param = 50
label_noise_type = "error"
dataset = "mnist"


""" Standard Classsifier """
# method = "StandardClassifier"
# grad_noise_stds = [0.01, 0.03, 0.1, 0.3]
# grad_noise_types = [None, 'Gaussian', 'Laplace']
#
# commands = []
# for n in ns:
#     for num_train_examples in n_samples_grid:
#         for grad_noise_type in grad_noise_types:
#             for grad_noise_std in grad_noise_stds:
#                 for seed in seeds:
#
#                     if grad_noise_type is not None:
#                         noisy_grad_name_string = f"-noisygrad-{grad_noise_type}-std{grad_noise_std}"
#                         noisy_grad_run_string = f"--add_noise --noise_type {grad_noise_type} --noise_std {grad_noise_std}"
#                     else:
#                         noisy_grad_name_string = ""
#                         noisy_grad_run_string = ""
#                         if grad_noise_std != grad_noise_stds[0]:
#                             continue
#
#                     if num_train_examples is not None:
#                         num_train_examples_name_string = f"-num_train_examples{num_train_examples}"
#                         num_train_examples_run_string = f"--num_train_examples {num_train_examples}"
#                     else:
#                         num_train_examples_name_string = ""
#                         num_train_examples_run_string = ""
#
#                     command = f"python -um scripts.train_classifier -c configs/4layer-cnn-mnist.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} {num_train_examples_run_string} --label_noise_type {label_noise_type} -m {method} {noisy_grad_run_string} --seed {seed} -l logs/{dataset}{num_train_examples_name_string}-{label_noise_type}-noise{n}-{method}{noisy_grad_name_string}-seed{seed}"
#                     commands += process_command(command)
#
# # Mean absolute error
# for n in ns:
#     for num_train_examples in n_samples_grid:
#         for seed in seeds:
#
#             if num_train_examples is not None:
#                 num_train_examples_name_string = f"-num_train_examples{num_train_examples}"
#                 num_train_examples_run_string = f"--num_train_examples {num_train_examples}"
#             else:
#                 num_train_examples_name_string = ""
#                 num_train_examples_run_string = ""
#
#             command = f"python -um scripts.train_classifier -c configs/4layer-cnn-mnist.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} {num_train_examples_run_string} --label_noise_type {label_noise_type} -m {method} --loss_function mae --seed {seed} -l logs/{dataset}{num_train_examples_name_string}-{label_noise_type}-noise{n}-{method}-mae-seed{seed}"
#             commands += process_command(command)
#
# merge_commands(commands, gpu_cnt=10)


""" FW """
# method = "StandardClassifier"
#
# commands = []
# for n in ns:
#     for num_train_examples in n_samples_grid:
#         for seed in seeds:
#
#             if num_train_examples is not None:
#                 num_train_examples_name_string = f"-num_train_examples{num_train_examples}"
#                 num_train_examples_run_string = f"--num_train_examples {num_train_examples}"
#             else:
#                 num_train_examples_name_string = ""
#                 num_train_examples_run_string = ""
#
#             load_from = f"logs/{dataset}{num_train_examples_name_string}-{label_noise_type}-noise{n}-{method}-seed{seed}/checkpoints/best_val.mdl"
#             command = f"python -um scripts.train_classifier -c configs/4layer-cnn-mnist.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} {num_train_examples_run_string} --label_noise_type {label_noise_type} -m {method} --loss_function fw --load_from {load_from} --seed {seed} -l logs/{dataset}{num_train_examples_name_string}-{label_noise_type}-noise{n}-{method}-fw-seed{seed}"
#             commands += process_command(command)
#
# # merge_commands(commands, gpu_cnt=10)


""" DMI """
# method = "StandardClassifier"
# lrs = [1e-3, 1e-4, 1e-5, 1e-6]
#
# # commands = []
# for n in ns:
#     for num_train_examples in n_samples_grid:
#         for lr in lrs:
#             for seed in seeds:
#
#                 if num_train_examples is not None:
#                     num_train_examples_name_string = f"-num_train_examples{num_train_examples}"
#                     num_train_examples_run_string = f"--num_train_examples {num_train_examples}"
#                 else:
#                     num_train_examples_name_string = ""
#                     num_train_examples_run_string = ""
#
#                 load_from = f"logs/{dataset}{num_train_examples_name_string}-{label_noise_type}-noise{n}-{method}-seed{seed}/checkpoints/best_val.mdl"
#                 command = f"python -um scripts.train_classifier -c configs/4layer-cnn-mnist.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} {num_train_examples_run_string} --label_noise_type {label_noise_type} -m {method} --loss_function dmi --lr {lr} --load_from {load_from} --seed {seed} -l logs/{dataset}{num_train_examples_name_string}-{label_noise_type}-noise{n}-{method}-dmi-lr{lr}-seed{seed}"
#                 commands += process_command(command)
#
# merge_commands(commands, gpu_cnt=10)


""" PredictGradOutput """
# method = "PredictGradOutput"
# stopping_param = 100
# Ls = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
# q_dists = ['Gaussian', 'Laplace']
#
# commands = []
# for n in ns:
#     for num_train_examples in n_samples_grid:
#         for L in Ls:
#             for q_dist in q_dists:
#                 for seed in seeds:
#
#                     if num_train_examples is not None:
#                         num_train_examples_name_string = f"-num_train_examples{num_train_examples}"
#                         num_train_examples_run_string = f"--num_train_examples {num_train_examples}"
#                     else:
#                         num_train_examples_name_string = ""
#                         num_train_examples_run_string = ""
#
#                     command = f"python -um scripts.train_classifier -c configs/4layer-cnn-mnist.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} {num_train_examples_run_string} --label_noise_type {label_noise_type} -m {method} --q_dist {q_dist} -L {L} --seed {seed} -l logs/{dataset}{num_train_examples_name_string}-{label_noise_type}-noise{n}-{method}-{q_dist}-L{L}-seed{seed}"
#                     commands += process_command(command)
#
# merge_commands(commands)


""" PredictGradOutput with sampling """
# method = "PredictGradOutput"
# stopping_param = 100
# Ls = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
# q_dists = ['Gaussian', 'Laplace']
# lambs = {
#     'Gaussian': [5.0, 50.0, 500.0, 5000.0],
#     'Laplace': [5, 10, 30, 100]
# }
#
# commands = []
# for n in ns:
#     for num_train_examples in n_samples_grid:
#         for L in Ls:
#             for q_dist in q_dists:
#                 for lamb in lambs[q_dist]:
#                     for seed in seeds[:3]:
#
#                         if num_train_examples is not None:
#                             num_train_examples_name_string = f"-num_train_examples{num_train_examples}"
#                             num_train_examples_run_string = f"--num_train_examples {num_train_examples}"
#                         else:
#                             num_train_examples_name_string = ""
#                             num_train_examples_run_string = ""
#
#                         command = f"python -um scripts.train_classifier -c configs/4layer-cnn-mnist.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} {num_train_examples_run_string} --label_noise_type {label_noise_type} -m {method} --q_dist {q_dist} --lamb {lamb} -L {L} --sample_from_q --seed {seed} -l logs/{dataset}{num_train_examples_name_string}-{label_noise_type}-noise{n}-{method}-{q_dist}-lamb{lamb}-L{L}-sample-seed{seed}"
#                         commands += process_command(command)
#
# merge_commands(commands, gpu_cnt=10, max_job_cnt=120)



""" PenalizeLastLayerFixedForm """
# method = "PenalizeLastLayerFixedForm"
# Ls = [0.0, 0.01, 0.1, 1.0, 10.0]
# lambs = [0.001, 0.01, 0.03, 0.1]
#
# commands = []
# for n in ns:
#     for num_train_examples in n_samples_grid:
#         for L in Ls:
#             for lamb in lambs:
#                 for seed in seeds[:3]:
#
#                     if num_train_examples is not None:
#                         num_train_examples_name_string = f"-num_train_examples{num_train_examples}"
#                         num_train_examples_run_string = f"--num_train_examples {num_train_examples}"
#                     else:
#                         num_train_examples_name_string = ""
#                         num_train_examples_run_string = ""
#
#                     command = f"python -um scripts.train_classifier -c configs/4layer-cnn-mnist.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} {num_train_examples_run_string} --label_noise_type {label_noise_type} -m {method} --lamb {lamb} -L {L} --seed {seed} -l logs/{dataset}{num_train_examples_name_string}-{label_noise_type}-noise{n}-{method}-lamb{lamb}-L{L}-seed{seed}"
#                     commands += process_command(command)
#
# merge_commands(commands)





########################################################################################################################
######################                            CIFAR10-error                                   ######################
########################################################################################################################
ns = [0.2, 0.4, 0.6, 0.8]
seeds = range(42, 43)
device = 'cuda'
n_epochs = 400
save_iter = 10000
vis_iter = 50
stopping_param = 100
label_noise_type = "error"
dataset = "cifar10"

""" Standard Classsifier """
# method = "StandardClassifier"
#
# commands = []
# for n in ns:
#     for seed in seeds:
#         command = f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -A -n {n} --label_noise_type {label_noise_type} -m {method} --seed {seed} -l logs/{dataset}-{label_noise_type}-noise{n}-augment-{method}-seed{seed}"
#         commands += process_command(command)
#
# # Mean absolute error
# for n in ns:
#     for seed in seeds:
#         command = f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -A -n {n} --label_noise_type {label_noise_type} -m {method} --loss_function mae --seed {seed} -l logs/{dataset}-{label_noise_type}-noise{n}-augment-{method}-mae-seed{seed}"
#         commands += process_command(command)
#
# # merge_commands(commands, gpu_cnt=10)



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



""" FW """
# method = "StandardClassifier"
#
# commands = []
# for n in ns:
#     for seed in seeds:
#         load_from = f"logs/{dataset}-{label_noise_type}-noise{n}-augment-{method}-seed{seed}/checkpoints/best_val.mdl"
#         command = f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -A -n {n} --label_noise_type {label_noise_type} -m {method} --loss_function fw --load_from {load_from} --seed {seed} -l logs/{dataset}-{label_noise_type}-noise{n}-augment-{method}-fw-seed{seed}"
#         commands += process_command(command)
#
# # merge_commands(commands, gpu_cnt=10)


""" DMI """
# method = "StandardClassifier"
# lrs = [1e-3, 1e-4, 1e-5, 1e-6]
#
# # commands = []
# for n in ns:
#     for lr in lrs:
#         for seed in seeds:
#             load_from = f"logs/{dataset}-{label_noise_type}-noise{n}-augment-{method}-seed{seed}/checkpoints/best_val.mdl"
#             command = f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -A -n {n} --label_noise_type {label_noise_type} -m {method} --loss_function dmi --lr {lr} --load_from {load_from} --seed {seed} -l logs/{dataset}-{label_noise_type}-noise{n}-augment-{method}-dmi-lr{lr}-seed{seed}"
#             commands += process_command(command)
#
# merge_commands(commands, gpu_cnt=10)


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



# ########################################################################################################################
# ######################                               CIFAR10-custom                                 ######################
# ########################################################################################################################

""" Standard """
# method = "StandardClassifier"
# seeds = range(42, 43)
# ns = [0.0, 0.1, 0.2, 0.3, 0.4]
# device = 'cuda'
#
# commands = []
# for n in ns:
#     for seed in seeds:
#         commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d cuda -e 400 -s 50 -v 50 -D cifar10 -n {n} --label_noise_type cifar10_custom -m {method} --seed {seed} -A -l logs/cifar10-custom-noise{n}-{method}-augment-seed{seed}")
#         commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d cuda -e 400 -s 50 -v 50 -D cifar10 -n {n} --label_noise_type cifar10_custom -m {method} --seed {seed} --loss_function mae -A -l logs/cifar10-custom-noise{n}-{method}-mae-augment-seed{seed}")
#
# merge_commands(commands)


""" DMI """
# method = "StandardClassifier"
# seeds = range(42, 43)  # range(42, 47)
# # ns = [0.1, 0.2, 0.3, 0.4]
# ns = [0.0]
# lrs = [1e-3, 1e-4, 1e-5, 1e-6]
# device = 'cuda'
#
# commands = []
# for n in ns:
#     for lr in lrs:
#         for seed in seeds:
#             commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d cuda -e 400 -s 50 -v 50 -D cifar10 -n {n} --label_noise_type cifar10_custom -m {method} --loss_function dmi --lr {lr} --stopping_param 100 --seed {seed} -A --load_from logs/cifar10-custom-noise{n}-{method}-augment-seed{seed}/checkpoints/best_val.mdl -l logs/cifar10-custom-noise{n}-{method}-dmi-lr{lr}-augment-seed{seed}")
#
# merge_commands(commands, gpu_cnt=10, max_job_cnt=2)


""" FW """
# method = "StandardClassifier"
# seeds = range(42, 43)  # range(42, 47)
# # ns = [0.1, 0.2, 0.3, 0.4]
# ns = [0.0]
# device = 'cuda'
#
# commands = []
# for n in ns:
#     for seed in seeds:
#         commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d cuda -e 400 -s 50 -v 50 -D cifar10 -n {n} --label_noise_type cifar10_custom -m {method} --loss_function fw --seed {seed} -A --load_from logs/cifar10-custom-noise{n}-{method}-augment-seed{seed}/checkpoints/best_val.mdl -l logs/cifar10-custom-noise{n}-{method}-fw-augment-seed{seed}")
#
# merge_commands(commands)


""" Predict Laplace """
# method = "PredictGradOutput"
# seeds = range(42, 43)
# Ls = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
# ns = [0.0, 0.1, 0.2, 0.3, 0.4]
# device = 'cuda'
#
# commands = []
# for n in ns:
#     for L in Ls:
#         for seed in seeds:
#             commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d cuda -e 400 --stopping_param 100 -s 50 -v 50 -D cifar10 -n {n} --label_noise_type cifar10_custom -m {method} --q_dist Laplace -L {L} --seed {seed} -A -l logs/cifar10-custom-noise{n}-{method}-Laplace-L{L}-augment-seed{seed}")
#
# # merge_commands(commands, gpu_cnt=10, max_job_cnt=2)

""" Predict Gaussian """
# method = "PredictGradOutput"
# seeds = range(42, 43)
# Ls = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
# ns = [0.0, 0.1, 0.2, 0.3, 0.4]
# device = 'cuda'
#
# # commands = []
# for n in ns:
#     for L in Ls:
#         for seed in seeds:
#             commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d cuda -e 400 -s 50 -v 50 -D cifar10 -n {n} --label_noise_type cifar10_custom -m {method} --q_dist Gaussian -L {L} --seed {seed} --stopping_param 100 -A -l logs/cifar10-custom-noise{n}-{method}-Gaussian-L{L}-augment-seed{seed}")
#
# # merge_commands(commands, gpu_cnt=10, max_job_cnt=2)


""" Predict Laplace [loaded, larger stopping param] """
# method = "PredictGradOutput"
# seeds = range(42, 43)
# Ls = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
# ns = [0.0, 0.1, 0.2, 0.3, 0.4]
# device = 'cuda'
#
# # commands = []
# for n in ns:
#     for L in Ls:
#         for seed in seeds:
#             commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d cuda -e 400 -s 50 -v 50 -D cifar10 -n {n} --label_noise_type cifar10_custom -m {method} --q_dist Laplace -L {L} --seed {seed} -A -l logs/cifar10-custom-noise{n}-{method}-Laplace-L{L}-augment-loaded-seed{seed} --stopping_param 200 --load_from logs/cifar10-custom-noise{n}-StandardClassifier-augment-seed{seed}/checkpoints/best_val.mdl")
#
# # merge_commands(commands, gpu_cnt=10, max_job_cnt=2)


""" Predict Gaussian [loaded] """
# method = "PredictGradOutput"
# seeds = range(42, 43)
# Ls = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
# ns = [0.0, 0.1, 0.2, 0.3, 0.4]
# device = 'cuda'
#
# # commands = []
# for n in ns:
#     for L in Ls:
#         for seed in seeds:
#             commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar10.json -d cuda -e 400 -s 200 -v 50 -D cifar10 -n {n} --label_noise_type cifar10_custom -m {method} --q_dist Gaussian -L {L} --seed {seed} --stopping_param 200 -A -l logs/cifar10-custom-noise{n}-{method}-Gaussian-L{L}-augment-loaded-seed{seed} --load_from logs/cifar10-custom-noise{n}-StandardClassifier-augment-seed{seed}/checkpoints/best_val.mdl")
#
# merge_commands(commands, gpu_cnt=10, max_job_cnt=2)



########################################################################################################################
######################                                  CIFAR-100                                  #####################
########################################################################################################################
# """ Standard """
# method = "StandardClassifier"
# seeds = range(42, 43)
# ns = [0.4]
# device = 'cuda'
#
# commands = []
# for n in ns:
#     for seed in seeds:
#         commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar100.json -d cuda -e 400 --stopping_param 100 -s 50 -v 50 -D cifar100 --label_noise_type error -n {n} -m {method} --seed {seed} -A -l logs/cifar100-error-noise{n}-{method}-augment-seed{seed}")
#         commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar100.json -d cuda -e 400 --stopping_param 100 -s 50 -v 50 -D  --label_noise_type error -n {n} -m {method} --seed {seed} --loss_function mae -A -l logs/cifar100-error-noise{n}-{method}-mae-augment-seed{seed}")
#
# merge_commands(commands)


""" DMI """
# method = "StandardClassifier"
# seeds = range(42, 43)
# ns = [0.4]
# lrs = [1e-3, 1e-4, 1e-5, 1e-6]
# device = 'cuda'
#
# commands = []
# for n in ns:
#     for lr in lrs:
#         for seed in seeds:
#             commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar100.json -d cuda -e 400 --stopping_param 100 -s 50 -v 50 -D cifar100 --label_noise_type error -n {n} -m {method} --loss_function dmi --lr {lr} --seed {seed} -A -l logs/cifar100-error-noise{n}-{method}-dmi-lr{lr}-augment-seed{seed} --load_from logs/cifar100-error-noise{n}-{method}-augment-seed{seed}/checkpoints/best_val.mdl")
#
# merge_commands(commands, gpu_cnt=10, max_job_cnt=2)


""" FW """
# method = "StandardClassifier"
# seeds = range(42, 43)
# ns = [0.4]
# device = 'cuda'
#
# commands = []
# for n in ns:
#     for seed in seeds:
#         commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar100.json -d cuda -e 400 --stopping_param 100 -s 50 -v 50 -D cifar100 --label_noise_type error -n {n} -m {method} --loss_function fw --seed {seed} -A -l logs/cifar100-error-noise{n}-{method}-fw-augment-seed{seed} --load_from logs/cifar100-error-noise{n}-{method}-augment-seed{seed}/checkpoints/best_val.mdl")
#
# merge_commands(commands, gpu_cnt=10, max_job_cnt=2)


""" Predict Laplace [custom noise, loaded, larger stopping param] """
# method = "PredictGradOutput"
# seeds = range(42, 43)  # range(42, 47)
# Ls = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
# ns = [0.4]
# device = 'cuda'
#
# commands = []
# for n in ns:
#     for L in Ls:
#         for seed in seeds:
#             commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar100.json -d cuda -e 400 -s 50 -v 50 -D cifar100 --label_noise_type error -n {n} -m {method} --q_dist Laplace -L {L} --seed {seed} -A -l logs/cifar100-error-noise{n}-{method}-Laplace-L{L}-augment-loaded-seed{seed} --stopping_param 200 --load_from logs/cifar100-error-noise{n}-StandardClassifier-augment-seed{seed}/checkpoints/best_val.mdl")
#
# merge_commands(commands, gpu_cnt=10, max_job_cnt=2)


""" Predict Gaussian [custom noise, loaded, larger stopping param] """
# method = "PredictGradOutput"
# seeds = range(42, 43)  # range(42, 47)
# Ls = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
# ns = [0.4]
# device = 'cuda'
#
# commands = []
# for n in ns:
#     for L in Ls:
#         for seed in seeds:
#             commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar100.json -d cuda -e 400 -s 500 -v 50 -D cifar100 --label_noise_type error -n {n} -m {method} --q_dist Gaussian -L {L} --seed {seed} -A -l logs/cifar100-error-noise{n}-{method}-Gaussian-L{L}-augment-loaded-seed{seed} --stopping_param 200 --load_from logs/cifar100-error-noise{n}-StandardClassifier-augment-seed{seed}/checkpoints/best_val.mdl")
#
# merge_commands(commands, gpu_cnt=3, max_job_cnt=2000, put_device_id=True)

""" Predict Laplace [custom noise, larger stopping param] """
# method = "PredictGradOutput"
# seeds = range(42, 43)
# Ls = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
# ns = [0.4]
# device = 'cuda'
#
# commands = []
# for n in ns:
#     for L in Ls:
#         for seed in seeds:
#             commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar100.json -d cuda -e 400 -s 50 -v 50 -D cifar100 --label_noise_type error -n {n} -m {method} --q_dist Laplace -L {L} --seed {seed} -A -l logs/cifar100-error-noise{n}-{method}-Laplace-L{L}-augment-seed{seed} --stopping_param 200")
#
# # merge_commands(commands, gpu_cnt=10, max_job_cnt=2)


""" Predict Gaussian [custom noise, larger stopping param] """
# method = "PredictGradOutput"
# seeds = range(42, 43)
# Ls = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
# ns = [0.4]
# device = 'cuda'
#
# # commands = []
# for n in ns:
#     for L in Ls:
#         for seed in seeds:
#             commands += process_command(f"python -um scripts.train_classifier -c configs/double-resnet-cifar100.json -d cuda -e 400 -s 500 -v 50 -D cifar100 --label_noise_type error -n {n} -m {method} --q_dist Gaussian -L {L} --seed {seed} -A -l logs/cifar100-error-noise{n}-{method}-Gaussian-L{L}-augment-seed{seed} --stopping_param 200")
#
# merge_commands(commands, gpu_cnt=3, max_job_cnt=2000, put_device_id=True)



########################################################################################################################
######################                          Preventing memorization                            #####################
########################################################################################################################
""" Standard """
method = "StandardClassifier"
# seeds = range(42, 47)
seeds = range(42, 43)
ns = [0.8]
device = 'cuda'
n_epochs = 400
save_iter = 10000
vis_iter = 50
stopping_param = n_epochs + 1
label_noise_type = "error"
dataset = "mnist"

""" no regularization """
commands = []
for n in ns:
    for seed in seeds:
        command = f"python -um scripts.train_classifier -c configs/4layer-mlp-mnist.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} --label_noise_type {label_noise_type} -m {method} --seed {seed} --clean_validation -l logs/prevent_overfitting_4layer/{dataset}-{label_noise_type}-noise{n}-{method}-seed{seed}"
        commands += process_command(command)
# merge_commands(commands, gpu_cnt=3, put_device_id=True)


""" dropout """
dropout_rates = [0.25, 0.5, 0.75]
# commands = []
for n in ns:
    for dropout_rate in dropout_rates:
        for seed in seeds:
            command = f"python -um scripts.train_classifier -c configs/4layer-mlp-mnist-dropout{dropout_rate}.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} --label_noise_type {label_noise_type} -m {method} --seed {seed} --clean_validation -l logs/prevent_overfitting_4layer/{dataset}-{label_noise_type}-noise{n}-{method}-dropout{dropout_rate}-seed{seed}"
            commands += process_command(command)
# merge_commands(commands, gpu_cnt=3, put_device_id=True)


""" weight decay """
weight_decay_rates = [0.001, 0.003, 0.0001]
# commands = []
for n in ns:
    for weight_decay in weight_decay_rates:
        for seed in seeds:
            command = f"python -um scripts.train_classifier -c configs/4layer-mlp-mnist.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} --label_noise_type {label_noise_type} -m {method} --weight_decay {weight_decay} --seed {seed} --clean_validation -l logs/prevent_overfitting_4layer/{dataset}-{label_noise_type}-noise{n}-{method}-weight_decay{weight_decay}-seed{seed}"
            commands += process_command(command)
# merge_commands(commands, gpu_cnt=3, put_device_id=True)


""" predict Laplace """
method = "PredictGradOutput"
Ls = [1.0, 3.0, 10.0, 30.0, 100.0]
# commands = []
for n in ns:
    for L in Ls:
        for seed in seeds:
            command = f"python -um scripts.train_classifier -c configs/4layer-mlp-mnist.json -d {device} -e {n_epochs} -s {save_iter} -v {vis_iter} --stopping_param {stopping_param} -D {dataset} -n {n} --label_noise_type {label_noise_type} -m {method} --q_dist Laplace --grad_weight_decay {L} --seed {seed} --clean_validation -l logs/prevent_overfitting_4layer/{dataset}-{label_noise_type}-noise{n}-{method}-Laplace-L{L}-seed{seed}"
            commands += process_command(command)
merge_commands(commands, gpu_cnt=3, put_device_id=True)
