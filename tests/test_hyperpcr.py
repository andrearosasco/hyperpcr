from pcr import __version__
from pcr.model.PCRNetwork import PCRNetwork
from pcr.pcn_training_config import Config
from pcr.misc import download_checkpoint
import torch


def test_version():
    assert __version__ == '0.1.0'

    model = PCRNetwork(Config.Model)
    ckpt_path = './checkpoint'
    download_checkpoint('1vAxN2MF7sWayeG1uvh2Jc6Gr4WHkCX6X', ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))

    partial = torch.rand((10000, 3))
    complete = model(partial)
    print(complete)

if __name__ == '__main__':
    test_version()