import requests
# import torch
# import os
# import wandb
# from pathlib import Path


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
#
#
# import requests

def download_checkpoint(id, destination):
    URL = f'https://drive.google.com/u/0/uc?id={id}&export=download'

    session = requests.Session()

    response = session.get(URL, stream = True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == '__main__':
    download_checkpoint('1vAxN2MF7sWayeG1uvh2Jc6Gr4WHkCX6X', './checkpoint')