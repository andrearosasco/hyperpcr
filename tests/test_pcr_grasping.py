import numpy as np
try:
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector
    from open3d.visualization import draw
    visualize = True
except ImportError:
    visualize = False

from pcr import __version__
from pcr.model import PCRNetwork as Model
from pcr.utils import Normalize, Denormalize
from pcr.pcn_training_config import Config
from pcr.misc import download_checkpoint
import torch


def test_version():
    assert __version__ == '0.1.0'

    ckpt_path = download_checkpoint(f'grasping.ckpt')
    asset_path = download_asset(f'partial_bleach_317.npy')

    model = Model.load_from_checkpoint(ckpt_path, config=Config.Model)
    model.cuda()
    model.eval()

    partial = np.load('../pcr/data/MCD/training_data/grasp_database/avocado_poisson_000/pointclouds/_0_0_0_partial.npy')

    partial, ctx = Normalize(Config.Processing)(partial)

    partial = torch.tensor(partial, dtype=torch.float32).cuda().unsqueeze(0)
    complete, probabilities = model(partial)

    complete = complete.squeeze(0).cpu().numpy()
    partial = partial.squeeze(0).cpu().numpy()

    complete = Denormalize(Config.Processing)(complete, ctx)
    partial = Denormalize(Config.Processing)(partial, ctx)

    if visualize:
        draw([
              PointCloud(points=Vector3dVector(partial)).paint_uniform_color([0, 0, 1]),
              PointCloud(points=Vector3dVector(complete)).paint_uniform_color([0, 1, 1]),
              ])

    print(complete)


if __name__ == '__main__':
    test_version()