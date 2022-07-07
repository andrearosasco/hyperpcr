from abc import ABC
from .Backbone import BackBone
from .Decoder import Decoder
from .ImplicitFunction import ImplicitFunction
import pytorch_lightning as pl


class PCRNetwork(pl.LightningModule, ABC):

    def __init__(self, config):
        super().__init__()
        self.backbone = BackBone(config)
        self.sdf = ImplicitFunction(config)
        self.decoder = Decoder(self.sdf, config.Decoder)

    def forward(self, partial):
        fast_weights, _ = self.backbone(partial)
        reconstruction, probabilities = self.decoder(fast_weights)

        return reconstruction, probabilities
