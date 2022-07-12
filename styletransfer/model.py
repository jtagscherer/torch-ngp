import logging

import torch
import torch.nn as nn

from styletransfer import RAIN
from styletransfer.RAIN import Net as RAIN_net

logger = logging.getLogger(__package__)


class StyleNeRFpp(nn.Module):
    def __init__(self,opt):
        super().__init__()

        self.latent_codes = nn.Embedding(1, 64).cuda()
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        # Create vgg and fc_encoder in RAIN_net
        vgg = RAIN.vgg
        fc_encoder = RAIN.fc_encoder

        # Load data weights of vgg and fc_encoder
        vgg.load_state_dict(torch.load(f'{opt.vgg_path}/vgg_normalised.pth'))
        fc_encoder.load_state_dict(torch.load(f'{opt.vgg_path}fc_encoder_iter_160000.pth'))

        vgg = nn.Sequential(*list(vgg.children())[:31])
        self.RAIN_net = RAIN_net(vgg, fc_encoder)

        # Fixed RAIN_net
        for param in self.RAIN_net.parameters():
            param.requires_grad = False

    def get_content_feat(self, content_img):
        return self.RAIN_net.get_content_feat(content_img)

    def get_style_feat(self, style_img):
        return self.RAIN_net.get_style_feat(style_img)
