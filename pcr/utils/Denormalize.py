import torch


class Denormalize:
    def __init__(self, config):
        self.scale = config.scale

    def __call__(self, pointcloud, context):
        pointcloud = (pointcloud + context['offset']) / self.scale * context['diameter'] + context['center']
        return pointcloud