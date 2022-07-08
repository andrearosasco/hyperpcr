import requests
import torch
from pathlib import Path
from rich.progress import track


def onnx_minimum(x1, x2):
    return torch.where(x2 < x1, x2, x1)


def fp_sampling(points, num: int):
    batch_size = points.shape[0]
    # TODO use onnx_cdists just to export to onnx, otherwise use torch.cdist
    # D = onnx_cdists(points, points)
    D = torch.cdist(points, points)
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    res = torch.zeros((batch_size, 1), dtype=torch.int32, device=points.device)
    ds = D[:, 0, :]
    for i in range(1, num):
        idx = ds.max(dim=1)[1]
        res = torch.cat([res, idx.unsqueeze(1).to(torch.int32)], dim=1)
        ds = onnx_minimum(ds, D[torch.arange(batch_size), idx, :])

    return res


# def download_checkpoint(id, version):
#     ckpt = f'model-{id}:{version}'
#     project = 'pcr-grasping'
#
#     ckpt_path = f'artifacts/{ckpt}/model.ckpt' if os.name != 'nt' else\
#                 f'artifacts/{ckpt}/model.ckpt'.replace(':', '-')
#
#     if not Path(ckpt_path).exists():
#         run = wandb.init(id=id, settings=wandb.Settings(start_method="spawn"))
#         run.use_artifact(f'rosasco/{project}/{ckpt}', type='model').download(f'artifacts/{ckpt}/')
#         wandb.finish(exit_code=0)
#
#     return ckpt_path
def download(url, dir, name):
    dir = Path(dir)
    file_path = dir / name
    full_url = f'{url}/{file_path.as_posix()}'

    if not dir.exists():
        dir.mkdir()

    if not file_path.exists():

        r = requests.get(full_url, stream=False)
        l = int(r.headers['Content-length']) / 1024

        with file_path.open('wb') as f:
            for chunk in track(r.iter_content(chunk_size=1024), total=l,
                               description=f'Downloading {dir.as_posix()[:-1]}...'):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    return file_path.as_posix()


def download_checkpoint(name):
    url = f'https://github.com/andrearosasco/hyperpcr/raw/main'
    dir = 'checkpoints'

    return download(url, dir, name)


def download_asset(name):
    url = f'https://github.com/andrearosasco/hyperpcr/raw/main'
    dir = 'assets'

    return download(url, dir, name)

