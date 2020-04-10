from modules import nn_utils, losses
from modules import visualization as vis
import torch
from nnlib.nnlib.utils import capture_arguments_of_init
from nnlib.nnlib.data_utils.base import revert_normalization


class VAE(torch.nn.Module):
    """ VAE with two additional regularization parameters.
     `beta`: weight of the KL term, as in beta-VAE
    """
    @capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, device='cuda', **kwargs):
        super(VAE, self).__init__()

        assert len(input_shape) == 3
        self.args = None  # this will be modified by the decorator
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args

        # used later
        self._vis_iters = 0

        # initialize the network
        self.hidden_shape = [None, self.architecture_args['hidden_dim']]

        self.decoder, _ = nn_utils.parse_feed_forward(args=self.architecture_args['decoder'],
                                                      input_shape=self.hidden_shape)
        self.decoder = self.decoder.to(device)

        self.encoder, _ = nn_utils.parse_feed_forward(args=self.architecture_args['encoder'],
                                                      input_shape=self.input_shape)

        self.encoder = self.encoder.to(device)

    def forward(self, inputs, sampling=False, detailed_output=False,
                grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        x = inputs[0].to(self.device)

        # z | x
        z_params = self.encoder(x)

        if sampling:
            z = self.encoder.sample(z_params)
        else:
            z = self.encoder.mean(z_params)

        # x | z
        x_rec = self.decoder(z)

        out = {
            'x_rec': x_rec,
            'z': z
        }

        # add z_params
        for k, v in z_params.items():
            out[k] = v

        # add input if needed
        if detailed_output:
            out['x'] = x

        return out

    def compute_loss(self, outputs, grad_enabled, dataset, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        eps = 1e-6
        target = torch.clamp(revert_normalization(outputs['x'], dataset=dataset), eps, 1 - eps)
        recon_loss = losses.binary_cross_entropy(target=target, pred=outputs['x_rec'])
        kl_loss = self.encoder.kl_divergence(outputs)

        batch_losses = {
            'recon': recon_loss,
            'kl': kl_loss
        }

        return batch_losses, outputs

    def visualize(self, train_loader, val_loader, **kwargs):
        self._vis_iters += 1
        visualizations = {}

        # add reconstruction plot
        if val_loader is not None:
            fig, _ = vis.reconstruction_plot(self, train_loader.dataset, val_loader.dataset)
            visualizations['reconstruction'] = fig

        # add manifold plot
        fig, _ = vis.manifold_plot(self, example_shape=self.input_shape[1:],
                                   low=-3, high=+3, n_points=10)
        visualizations['manifold'] = fig

        # scatter plot
        if val_loader is not None:
            fig, _ = vis.latent_scatter(self, val_loader)
            visualizations['scatter'] = fig

        # latent space T-SNE plot, plotted less often since it is slower
        if (val_loader is not None) and self._vis_iters % 5 == 0:
            fig, _ = vis.latent_space_tsne(self, val_loader)
            visualizations['latent space T-SNE'] = fig

        return visualizations
