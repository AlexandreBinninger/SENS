from utils import mesh_utils, files_utils, train_utils
from custom_types import *
from pointnetpp.pointnet_utils import PointNetSetAbstraction
from models import models_utils
import options
from pointnetpp.pointnet_msg import get_model_b


class get_model(nn.Module):
    def __init__(self,num_class, normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(nnf.relu(self.bn1(self.fc1(x))))
        x = self.drop2(nnf.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x) * .5
        # x = nnf.log_softmax(x, -1)
        return x.softmax(-1)


def sample_pc(mesh_path: str, num_points: int, with_normals: bool) -> T:
    vs, faces = files_utils.load_mesh(mesh_path)
    # vs[:, 2] = -vs[:, 2]
    mesh = vs, faces
    mesh = mesh_utils.to_unit_sphere(mesh, scale=.9)
    face_areas, face_normals = mesh_utils.compute_face_areas(mesh)
    pc, chosen_faces_inds, _ = mesh_utils.sample_on_mesh(mesh, num_points, face_areas=face_areas)
    if with_normals:
        normals = face_normals[chosen_faces_inds]
        pc = torch.cat((pc, normals), dim=1)
    return pc


class ShapeDs(Dataset):

    def get_name(self, item):
        return self.shapes_paths[item][1]

    def __getitem__(self, item):
        return sample_pc("".join(self.shapes_paths[item]), self.num_samples, self.with_normals), self.get_name(item)

    def __len__(self):
        return len(self.shapes_paths)

    def __init__(self, root: str, num_samples: int, with_normals: bool):
        self.shapes_paths = files_utils.collect(root, '.obj', )
        self.num_samples = num_samples
        self.with_normals = with_normals



def get_dkl(p: T, q: T) -> T:
    mask = p.eq(0)
    denom = p.clone()
    q = q.clone()
    q[mask] = 1.
    denom[mask] = 1.
    dkl = (p * torch.log(denom / q)).sum(-1)
    return dkl

def get_jensen_shannon_divergence(p: T, q: T):
    m = (p + q) / 2
    jsd = (get_dkl(p, m) + get_dkl(q, m)) / 2
    return jsd

def eval_area(root: str, name: str):

    def get_area(path_):
        try:
            mesh = files_utils.load_mesh(''.join(path_))
            area = mesh_utils.compute_face_areas(mesh)[0].sum()
        except:
            print("".join(path_))
            return None
        return area

    meshes_predict = files_utils.collect(f"{root}/eval_mix/{name}/", '.obj')
    mesh_gt = files_utils.collect(f"{root}/eval_mix/ref/", '.obj')
    area_predict = {path[1]: get_area(path) for path in meshes_predict}
    area_predict = {key: item for key, item in area_predict.items() if item is not None}
    area_gt = {path[1]: get_area(path) for path in mesh_gt}
    area_gt = {key: item for key,item in area_gt.items() if item is not None}
    names = [key for key in area_gt if key in area_predict]
    all_area_predict = torch.tensor([area_predict[name] for name in names])
    all_area_gt = torch.tensor([area_gt[name] for name in names])
    all_area_predict = all_area_predict / all_area_gt
    all_area_error = (all_area_predict - 1).abs()
    print(all_area_error.mean())
    return

