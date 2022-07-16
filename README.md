This repository contains the code relative to the paper "Towards Confidence-guided Shape Completion for Robotic Applications".
This code contain a python package that allows you to do inference with our model. If you want to look at the training code check out [https://github.com/andrearosasco/pcr](https://github.com/andrearosasco/pcr).
### Installation
To install the code run
`pip install hyperpcr`
If you are on Linux this should automatically install all the dependencies.
On windows you might need to manually install pytorch.
### Usage
The following script download a checkpoint and a partial point cloud. Then it process it and generates its reconstruction. If you wanna visualize the results you have to additionally install Open3D python library.
```python
import numpy as np
import torch
try:
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector
    from open3d.visualization import draw
    visualize = True
except ImportError:
    visualize = False

from pcr.model import PCRNetwork as Model
from pcr.utils import Normalize, Denormalize
from pcr.default_config import Config
from pcr.misc import download_checkpoint, download_asset


def main():

    ckpt_path = download_checkpoint(f'grasping.ckpt')
    asset_path = download_asset(f'partial_bleach_317.npy')

    model = Model(config=Config.Model)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model.cuda()
    model.eval()

    partial = np.load(asset_path)

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
```
If you want to change the Decoder behaviour you can change the configuration values of `pcr.default_config.Config`.
