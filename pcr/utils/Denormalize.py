import torch


class Denormalize:
    def __init__(self, config):
        self.scale = config.scale

    def __call__(self, pointcloud, context):
        pointcloud = (pointcloud + torch.tensor(context['offset'], device=pointcloud.device)) / self.scale * torch.tensor(context['diameter'], device=pointcloud.device) + torch.tensor(context['center'], device=pointcloud.device)
        return pointcloud