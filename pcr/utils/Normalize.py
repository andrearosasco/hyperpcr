import numpy as np


class Normalize:
    def __init__(self, config):
        self.use_offset: bool = config.use_offset
        self.scale: float = config.scale

    def __call__(self, pointcloud: np.ndarray, context=None):
        if context is None:
            center = get_bbox_center(pointcloud)
            diameter = get_diameter(pointcloud - center)

            offset = 0
            if self.use_offset:
                offset = np.array([0, 0, 0.5 - ((pointcloud - center) / diameter * self.scale)[
                    np.argmax(((pointcloud - center) / diameter * 0.7)[..., 2])][..., 2]])
        else:
            center, diameter, offset = context['center'], context['diameter'], context['offset']

        pointcloud = ((pointcloud - center) / diameter * self.scale) - offset

        choice = np.random.permutation(pointcloud.shape[0])
        pointcloud = pointcloud[choice[:2048]]

        if pointcloud.shape[0] < 2048:
            zeros = np.zeros((2048 - pointcloud.shape[0], 3))
            pointcloud = np.concatenate([pointcloud, zeros])

        context = {'center': center, 'diameter': diameter, 'offset': offset}

        return pointcloud, context


def get_bbox_center(pc):
    center = pc.min(0) + (pc.max(0) - pc.min(0)) / 2.0
    return center


def get_diameter(pc):
    diameter = np.max(pc.max(0) - pc.min(0))
    return diameter
