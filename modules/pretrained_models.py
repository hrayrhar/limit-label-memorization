import torch
import torch.nn.functional as F
from torchvision import models
from modules import utils


class PretrainedResNet34(torch.nn.Module):
    """ Pretrained ResNet34. Expects input of size 224x224 that is normalized with
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
    Returns activations of the last layer before the fc layer. Output size is 7x7x512.
    """
    def __init__(self):
        super(PretrainedResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.output_shape = [None, 512]

        # freeze weights
        params = dict(self.resnet.named_parameters())
        for name, param in self.resnet.named_parameters():
            params[name].requires_grad = False

    def forward(self, x):
        assert 3 % x.shape[1] == 0
        x = x.repeat_interleave(3 // x.shape[1], dim=1)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')

        # ResNet's forward function
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.resnet.fc(x)
        return x.reshape((x.shape[0], -1))


class PretrainedMNISTVAE(torch.nn.Module):
    def __init__(self, path, device):
        super(PretrainedMNISTVAE, self).__init__()
        self.vae = utils.load(path, device=device)
        self.output_shape = [None, 128]

        # freeze weights
        params = dict(self.vae.named_parameters())
        for name, param in self.vae.named_parameters():
            params[name].requires_grad = False

    def forward(self, x):
        # normalize x to make it live in [0, 1], because VAE was trained on such data
        mean = 0.456
        std = 0.224
        return self.vae.forward([mean + std * x], sampling=False,
                                grad_enabled=True)['z']


class Identity(torch.nn.Module):
    def __init__(self, input_shape):
        super(Identity, self).__int__()
        self.output_shape = input_shape

    def forward(self, x):
        return x


def get_pretrained_model(pretrained_arg, input_shape, device):
    if pretrained_arg is None:
        return Identity(input_shape).to(device)
    if pretrained_arg == 'resnet':
        return PretrainedResNet34().to(device)
    return PretrainedMNISTVAE(pretrained_arg, device)
