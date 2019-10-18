from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from modules import utils
from tqdm import tqdm
import os
import time
import numpy as np
import torch


def build_optimizer(params, optimization_args):
    args = optimization_args['optimizer']
    optimizer = None
    name = args.pop('name', 'adam')
    if name == 'adam':
        optimizer = optim.Adam(params, **args)
    if name == 'sgd':
        optimizer = optim.SGD(params, **args)
    return optimizer


def build_scheduler(optimizer, optimization_args):
    args = optimization_args.get('scheduler', {})
    step_size = args.get('step_size', 1)
    gamma = args.get('gamma', 1.0)
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def run_partition(model, epoch, tensorboard, optimizer, loader, partition, training):
    if hasattr(model, 'on_epoch_start'):
        model.on_epoch_start(epoch=epoch, tensorboard=tensorboard, partition=partition)

    losses = defaultdict(list)
    total_number_samples = 0

    for (batch_data, batch_labels) in tqdm(loader, desc='{} batches'.format(partition)):
        # make the input and labels lists
        if isinstance(batch_data, torch.Tensor):
            batch_data = [batch_data]
        if isinstance(batch_labels, torch.Tensor):
            batch_labels = [batch_labels]

        # zero gradients in training phase
        if training:
            optimizer.zero_grad()

        # forward pass
        batch_losses, info = model.compute_loss(inputs=batch_data, labels=batch_labels,
                                                grad_enabled=training)
        batch_total_loss = sum([loss for name, loss in batch_losses.items()])

        if training:
            # backward pass & update
            batch_total_loss.backward()
            optimizer.step()  # update the parameters

        # call methods on_iteration_end if it is present
        # the network can compute some metrics here
        if hasattr(model, 'on_iteration_end'):
            model.on_iteration_end(info=info, batch_losses=batch_losses, batch_labels=batch_labels,
                                   partition=partition, tensorboard=tensorboard)

        # collect all losses
        if len(batch_losses) > 1:
            batch_losses['total'] = batch_total_loss
        for k, v in batch_losses.items():
            losses['{}_{}'.format(partition, k)].append(len(batch_data) * utils.to_numpy(v))
        total_number_samples += len(batch_data)

    for k, v in losses.items():
        losses[k] = np.sum(v) / total_number_samples
        tensorboard.add_scalar('losses/{}'.format(k), losses[k], epoch)

    if hasattr(model, 'on_epoch_end'):
        model.on_epoch_end(epoch=epoch, tensorboard=tensorboard, partition=partition,
                           loader=loader)

    return losses


def train(model, train_loader, val_loader, epochs, save_iter=10, vis_iter=4,
          optimization_args=None, log_dir=None):
    """ Trains the model. Validation loader can be none.
    Assumptions:
    1. loaders return (batch_inputs, batch_labels), where both can be lists or torch.Tensors
    """
    # if log_dir is not given, logging will be done a new directory in 'logs/' directory
    if log_dir is None:
        log_root = 'logs/'
        utils.make_path(log_root)
        last_run = max([0] + [int(k) for k in os.listdir(log_root) if k.isdigit()])
        log_dir = os.path.join(log_root, '{0:04d}'.format(last_run + 1))
        utils.make_path(log_dir)

    tensorboard = SummaryWriter(log_dir)
    print("Visualize logs using: tensorboard --logdir={0}".format(log_dir))

    optimizer = build_optimizer(model.parameters(), optimization_args)
    scheduler = build_scheduler(optimizer, optimization_args)

    for epoch in range(epochs):
        t0 = time.time()

        model.train()
        train_losses = run_partition(model=model, epoch=epoch, tensorboard=tensorboard, optimizer=optimizer,
                                     loader=train_loader, partition='train', training=True)

        val_losses = {}
        if val_loader is not None:
            model.eval()
            val_losses = run_partition(model=model, epoch=epoch, tensorboard=tensorboard, optimizer=optimizer,
                                       loader=val_loader, partition='val', training=False)

        # log some statistics
        t = time.time()
        log_string = 'Epoch: {}/{}'.format(epoch, epochs)
        for k, v in list(train_losses.items()) + list(val_losses.items()):
            log_string += ', {}: {:0.6f}'.format(k, v)
        log_string += ', Time: {:0.1f}s'.format(t - t0)
        print(log_string)

        # add visualizations
        if (epoch + 1) % vis_iter == 0 and hasattr(model, 'visualize'):
            visualizations = model.visualize(train_loader, val_loader, tensorboard=tensorboard, epoch=epoch)
            # visualizations is a dictionary containing figures in (name, fig) format.
            # there are visualizations created using matplotlib rather than tensorboard
            for (name, fig) in visualizations.items():
                tensorboard.add_figure(name, fig, epoch)

        # save the model
        if (epoch + 1) % save_iter == 0:
            utils.save(model=model, optimizer=optimizer, scheduler=scheduler,
                       path=os.path.join(log_dir, 'checkpoints', 'epoch{}.mdl'.format(epoch)))

        # update the learning rate
        scheduler.step()

    # enable testing mode
    model.eval()

    # save the final version of the network
    utils.save(model=model, optimizer=optimizer, scheduler=scheduler,
               path=os.path.join(log_dir, 'checkpoints', 'final.mdl'))

