from pcr import __version__
from pcr.model.PCRNetwork import PCRNetwork
from pcr.pcn_training_config import Config
from pcr.misc import download_checkpoint
import torch


def test_version():
    assert __version__ == '0.1.0'

    id = '2kkeig53'
    version = 'v9'

    model = PCRNetwork(Config.Model)
    ckpt_path = download_checkpoint(id, version)
    model.load_state_dict(torch.load(ckpt_path))

    partial = torch.rand((10000, 3))
    complete = model(partial)
    print(complete)
