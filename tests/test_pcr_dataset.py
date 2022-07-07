import numpy as np
from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
from open3d.visualization import draw
from scipy.spatial.transform import Rotation

from pcr import __version__
from pcr.model import PCRNetwork as Model
from pcr.utils import Normalize, Denormalize
from pcr.pcn_training_config import Config
from pcr.misc import download_checkpoint
import torch


def test_version():
    assert __version__ == '0.1.0'
    Config.Processing.use_offset = False
    Config.Processing.scale = 1.0

    id = '1m5301hl'
    version = 'v53'
    ckpt_path = download_checkpoint(f'model-{id}-{version}.ckpt')

    model = Model.load_from_checkpoint(ckpt_path, config=Config.Model)
    model.cuda()
    model.eval()

    partial = np.load('../pcr/data/MCD/training_data/grasp_database/avocado_poisson_000/pointclouds/_0_0_0_partial.npy')
    triangles = np.load('../pcr/data/MCD/ground_truth_meshes/grasp_database/avocado_poisson_000/meshes/triangles.npy')
    vertices = np.load('../pcr/data/MCD/ground_truth_meshes/grasp_database/avocado_poisson_000/meshes/vertices.npy')
    y1_pose = np.load('../pcr/data/MCD/training_data/grasp_database/avocado_poisson_000/pointclouds/_0_0_0_model_pose.npy')
    x2_pose = np.load('../pcr/data/MCD/training_data/grasp_database/avocado_poisson_000/pointclouds/_0_0_0_model_pose.npy')
    y3_pose = np.load('../pcr/data/MCD/training_data/grasp_database/avocado_poisson_000/pointclouds/_0_0_0_model_pose.npy')

    vertices = match_mesh_to_partial(vertices, [y1_pose, x2_pose, y3_pose])

    complete = np.array(TriangleMesh(vertices=Vector3dVector(vertices), triangles=Vector3iVector(triangles))
                        .sample_points_uniformly(16000).points)

    partial = partial + np.array([0, 0, -1])

    _, ctx = Normalize(Config.Processing)(complete)
    normalized_partial, _ = Normalize(Config.Processing)(partial, ctx)

    draw([PointCloud(points=Vector3dVector(normalized_partial)).paint_uniform_color([1, 0, 0]),
          PointCloud(points=Vector3dVector(partial)).paint_uniform_color([0, 0, 1]),
          create_cube()])

    tensor_partial = torch.tensor(normalized_partial, dtype=torch.float32, device=Config.General.device).unsqueeze(0)

    normalized_complete, probabilities = model(tensor_partial)
    complete = Denormalize(Config.Processing)(normalized_complete, ctx)

    draw([PointCloud(points=Vector3dVector(normalized_partial)).paint_uniform_color([1, 0, 0]),
          PointCloud(points=Vector3dVector(partial)).paint_uniform_color([0, 0, 1]),
          PointCloud(points=Vector3dVector(normalized_complete[0].cpu().numpy())).paint_uniform_color([1, 1, 0]),
          PointCloud(points=Vector3dVector(complete[0].cpu().numpy())).paint_uniform_color([0, 1, 1]),
          create_cube()
          ])

    print(complete)

def create_cube():
    cube = []
    for _ in range(2500):
        p = np.random.rand((3)) - 0.5
        cube.append([-0.5, p[1], p[2]])
        cube.append([0.5, p[1], p[2]])
        cube.append([p[0], -0.5, p[2]])
        cube.append([p[0], 0.5, p[2]])
        cube.append([p[0], p[1], -0.5])
        cube.append([p[0], p[1], 0.5])

    cb = PointCloud()
    cb.points = Vector3dVector(np.array(cube))
    cb.paint_uniform_color([0, 1, 0])

    return cb

def correct_pose(pose):
    pre = Rotation.from_euler('yz', [90, 180], degrees=True).as_matrix()
    post = Rotation.from_euler('zyz', [180, 180, 90], degrees=True).as_matrix()

    t = np.eye(4)
    t[:3, :3] = np.dot(post, np.dot(pose[:3, :3].T, pre))

    return t


def get_x_rot(pose):
    t = correct_pose(pose)

    x, y, z = Rotation.from_matrix(t[:3, :3].T).as_euler('xyz', degrees=True)

    rot_x = np.eye(4)
    rot_x[:3, :3] = Rotation.from_euler('x', x, degrees=True).as_matrix() @ Rotation.from_euler('x', 180,
                                                                                                degrees=True).as_matrix()

    return rot_x


def get_y_rot(pose):
    t = correct_pose(pose)

    x, y, z = Rotation.from_matrix(t[:3, :3].T).as_euler('xyz', degrees=True)
    if z < 0:
        t[:3, :3] = Rotation.from_euler('zxy', [-z, -x, y], degrees=True).as_matrix()
    else:
        t[:3, :3] = Rotation.from_euler('zxy', [-z, x, -y], degrees=True).as_matrix()

    theta_y = Rotation.from_matrix(t[:3, :3]).as_euler('zxy', degrees=True)[2]
    rot_y = np.eye(4)
    rot_y[:3, :3] = Rotation.from_euler('y', theta_y, degrees=True).as_matrix()

    return rot_y


def match_mesh_to_partial(vertices, pose):
    y1_pose, x2_pose, y3_pose = pose

    y_rot = get_y_rot(y1_pose)
    x2_rot = get_x_rot(x2_pose)
    y3_rot = get_y_rot(y3_pose)

    base_rot = np.eye(4)
    base_rot[:3, :3] = Rotation.from_euler('xyz', [180, 0, -90], degrees=True).as_matrix()

    t = y_rot @ x2_rot @ y3_rot @ base_rot

    pc = np.ones((np.size(vertices, 0), 4))
    pc[:, 0:3] = vertices

    pc = pc.T
    pc = t @ pc
    pc = pc.T[..., :3]

    return pc

if __name__ == '__main__':
    test_version()