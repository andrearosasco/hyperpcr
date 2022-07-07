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
        self.decoder = Decoder(self.sdf, 8192*2, 0.7, 20)

    def forward(self, partial, object_id=None, step=0.01):
        fast_weights, _ = self.backbone(partial)
        pc, _ = self.decoder(fast_weights)

        return pc
