import torch
from .Backbone import BackBone
from .Decoder import Decoder
from .ImplicitFunction import ImplicitFunction


class PCRNetwork(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone = BackBone(config)
        self.sdf = ImplicitFunction(config)
        self.decoder = Decoder(self.sdf, config.Decoder)

    def forward(self, partial):
        fast_weights, _ = self.backbone(partial)
        reconstruction, probabilities = self.decoder(fast_weights)

        return reconstruction, probabilities
