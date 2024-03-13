from custom_types import *
from functools import reduce
from utils import mesh_utils
import abc


class GeometryDS(abc.ABC):

    @abc.abstractmethod
    def to(self, device: D):
        raise NotImplemented

    @property
    @abc.abstractmethod
    def vs_normals(self):
        raise NotImplemented

    @property
    @abc.abstractmethod
    def areas(self):
        raise NotImplemented


class MeshDS(GeometryDS):

    @property
    def face_angles(self) -> T:
        edge_vs = self.vs[self.edges]
        edge_vec = edge_vs[:, 0] - edge_vs[:, 1]
        lengths = edge_vec.norm(2, 1)
        lengths_next = lengths[self.next]
        edge_vec_next = -edge_vec[self.next]
        cos_angle = torch.einsum('ed,ed->e', edge_vec, edge_vec_next) / (lengths * lengths_next)
        cos_angle = cos_angle.clamp_(-1, 1)
        angles = torch.acos(cos_angle)
        return angles

    @property
    def edge_angles(self) -> T:
        normals_a = self.normals[self.e2f[self.unique_edges_id]]
        normals_b = self.normals[self.e2f[self.twins[self.unique_edges_id]]]
        cos_angles = torch.einsum('ed,ed->e', normals_a, normals_b)
        cos_angles = cos_angles.clamp_(-1, 1)
        angles = torch.acos(cos_angles) / np.pi
        return angles

    # @property
    # def edge_angles(self) -> T:
    #     normals_a = self.normals[self.e2f[self.unique_edges_id]]
    #     normals_b = self.normals[self.e2f[self.twins[self.unique_edges_id]]]
    #     edges = self.vs[self.unique_edges]
    #     edges = edges[:, 1] - edges[:, 0]
    #     edges = edges / edges.norm(2, dim=1)[:, None]
    #     dirs = edges.cross(normals_a), (-edges).cross(normals_b)
    #     check = (dirs[0] + dirs[1]) / 2
    #     cos_angles_check = torch.einsum('ed,ed->e', normals_a, check)
    #     cos_angles = torch.einsum('ed,ed->e', *dirs)
    #     cos_angles = cos_angles.clamp_(-1, 1)
    #     angles = torch.acos(cos_angles)
    #     # mask = cos_angles_check.gt(0).float()
    #     # angles = (2 * np.pi - angles) * mask + angles * (1 - mask)
    #     # angles[mask] = 2 * np.pi - angles[mask]
    #
    #     return angles

    @property
    def vs_angles(self) -> T:
        vs_angles = self.face_angles[self.v2e] * self.ring_mask.float()
        return vs_angles

    def edge_curvature(self) -> T:
        vs_edges = self.vs[self.edges]
        delta_vs = vs_edges[:, 1] - vs_edges[:, 0]
        # scale = delta_vs.norm(2, 1).mean()
        # delta_vs = delta_vs
        normals_edges = self.vs_normals[self.edges]
        delta_normals = normals_edges[:, 1] - normals_edges[:, 0]
        edge_curvature = torch.einsum('nd,nd->n', [delta_vs, delta_normals]) / (delta_vs ** 2).sum(1)
        return edge_curvature

    def simple_curvature(self):
        edge_curvature = self.edge_curvature()
        vs_curvature = edge_curvature[self.v2e] * self.ring_mask.float()
        return vs_curvature

    def simple_mean_curvature(self):
        vs_curvature = self.simple_curvature()
        mean_curvature = vs_curvature.sum(1) / self.vs_degree.float()
        return mean_curvature

    def simple_principle_curvature(self):
        raise NotImplemented

    @staticmethod
    def init_v_degree(mesh: T_Mesh) -> T:
        vs, faces = mesh
        vs_degree = torch.zeros(vs.shape[0], dtype=torch.int64)
        for face in faces:
            vs_degree[face] += 1
        return vs_degree

    def build_ds(self) -> TNS:

        def insert_face():
            new_edges = face[edge_extract]
            new_edges_id = face_id * 3 + zero_one_two
            # v2f[face, v2f_deg[face]] = face_id
            v2e[face, v2e_deg[face]] = new_edges_id
            v2e_deg[face] += 1
            edges[new_edges_id] = new_edges
            e2f[new_edges_id] = face_id
            next_[new_edges_id] = new_edges_id[one_two_zero]
            new_edges = new_edges.sort(1)[0]
            for i in range(3):
                edge = (new_edges[i, 0].item(), new_edges[i, 1].item())
                if edge in edge2key:
                    twins[edge2key[edge]] = new_edges_id[i]
                    twins[new_edges_id[i]] = edge2key[edge]
                else:
                    edge2key[edge] = new_edges_id[i]
                    unique_edges_id.append(new_edges_id[i])

        v2e = torch.zeros(len(self), self.max_degree, dtype=torch.int64) - 1
        v2side = torch.zeros(len(self), self.max_degree, dtype=torch.int64) - 1
        # v2f = torch.zeros(len(self), self.max_degree, dtype=torch.int64) - 1
        v2e_deg = torch.zeros(len(self), dtype=torch.int64)
        edges = torch.zeros(self.faces.shape[0] * 3, 2, dtype=torch.int64) - 1
        twins = torch.zeros(self.faces.shape[0] * 3, dtype=torch.int64) - 1
        e2f = torch.zeros(self.faces.shape[0] * 3, dtype=torch.int64)
        next_ = torch.zeros(self.faces.shape[0] * 3, dtype=torch.int64)
        zero_one_two = torch.arange(3)
        one_two_zero = (zero_one_two + 1) % 3
        edge_extract = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int64)
        unique_edges_id = []
        edge2key = dict()
        for face_id, face in enumerate(self.faces):
            insert_face()
        unique_edges_id = torch.tensor(unique_edges_id, dtype=torch.int64)
        return v2e, e2f, edges, twins, next_, unique_edges_id

    @property
    def unique_edges(self) -> T:
        return self.edges[self.unique_edges_id]

    def __len__(self):
        return self.vs.shape[0]

    @property
    def v2v(self):
        return self.edges[self.v2e][:, :, 1]

    @property
    def vs(self):
        return self.mesh[0]

    @vs.setter
    def vs(self, vs):
        self.invalidate()
        self.mesh = (vs, self.mesh[1])

    def get_mesh(self) -> T_Mesh:
        return self.vs.clone(), self.faces.clone()

    @property
    def faces(self):
        return self.mesh[1]

    @property
    def area_weights(self):
        return self.areas[self.v2f] * self.ring_mask.float()

    @property
    def is_manifold(self) -> bool:
        return self.edges.ne(-1).all().item()

    @property
    def is_closed(self) -> bool:
        return self.twins.ne(-1).all().item()

    @property
    def genus(self) -> int:
        return int(self.faces.shape[0] / 4 - self.vs.shape[0] / 2) + 1

    @property
    def vs_normals(self) -> T:
        vs_normals = torch.einsum('nrd,nr->nd', self.normals[self.v2f], self.area_weights)
        vs_normals = vs_normals / vs_normals.norm(2, 1)[:, None]
        return vs_normals

    @property
    def normals(self) -> T:
        if self.areas_normals_ is None:
            self.areas_normals_ = mesh_utils.compute_face_areas(self.mesh)
        return self.areas_normals_[1]

    @property
    def areas(self):
        if self.areas_normals_ is None:
            self.areas_normals_ = mesh_utils.compute_face_areas(self.mesh)
        return self.areas_normals_[0]

    def to(self, device: D):
        self.mesh = mesh_utils.to(self.mesh, device)
        self.ring_mask, self.vs_degree = mesh_utils.to((self.ring_mask, self.vs_degree), device)
        self.v2e, self.e2f, self.edges = mesh_utils.to((self.v2e, self.e2f, self.edges), device)
        self.twins, self.next = mesh_utils.to((self.twins, self.next), device)
        self.unique_edges_id, self.v2f_, self.areas_normals_ = mesh_utils.to((self.unique_edges_id, self.v2f_,
                                                                              self.areas_normals_), device)
        return self

    def unroll_edge_fe(self, unique_fe: T):
        to_squeeze = unique_fe.dim() == 1
        if to_squeeze:
            unique_fe = unique_fe.unsqueeze(1)
        fe = torch.zeros(self.edges.shape[0], unique_fe.shape[1], device=unique_fe.device)
        fe[self.unique_edges_id] = unique_fe
        fe[self.twins[self.unique_edges_id]] = unique_fe
        if to_squeeze:
            fe = fe.squeeze_(1)
        return fe

    @property
    def edge_lengths(self) -> T:
        edge_vs = self.vs[self.unique_edges]
        lengths = (edge_vs[:, 0] - edge_vs[:, 1]).norm(2, 1)
        return lengths

    def simple_laplace(self) -> T:
        vs_ring = self.vs[self.v2v] * self.ring_mask[:, :, None].float()
        ring_center = vs_ring.sum(1) / self.vs_degree[:, None].float()
        ring_center[torch.isnan(ring_center)] = 0
        diff = self.vs - ring_center
        return diff

    def graph_laplacian(self):
        arange = torch.arange(len(self), device=self.device)
        gl = torch.eye(len(self) + 1, len(self) + 1, device=self.device)
        gl[arange, arange] = self.vs_degree.float()
        gl = gl.flatten()
        vs_ring = self.edges[self.v2e][:, :, 1]
        vs_ring += (arange * (len(self) + 1))[:, None]
        vs_ring = vs_ring.flatten()
        vs_ring[~self.ring_mask.flatten()] = -1
        gl[vs_ring] = -1
        gl = gl.view(len(self) + 1, len(self) + 1)[:len(self), :len(self)]
        return gl

    @property
    def adjacency(self) -> T:
        adj = torch.zeros(len(self), len(self), device=self.device, dtype=self.vs.dtype)
        edge_lengths = self.edge_lengths
        edges = self.unique_edges
        adj[edges[:, 0], edges[:, 1]] = edge_lengths
        adj[edges[:, 1], edges[:, 0]] = edge_lengths
        return adj

    @property
    def device(self) -> D:
        return self.vs.device

    @property
    def v2f(self) -> T:
        if self.v2f_ is None:
            v2f = self.e2f[self.v2e]
            v2f[self.v2e.eq(-1)] = -1
            self.v2f_ = v2f
        return self.v2f_
        # in place

    def invalidate(self):
        self.areas_normals_ = None

    def sort_edges(self):
        def swap():  # swapping faces, edges, all_faces
            hold = self.v2e[relevant_rows, i]
            self.v2e[relevant_rows, i] = self.v2e[relevant_rows][ma]
            self.v2e[relevant_rows, where] = hold
            hold = searching_edges[relevant_rows, i]
            searching_edges[relevant_rows, i] = searching_edges[relevant_rows][ma]
            searching_edges[relevant_rows, where] = hold

        self.ring_mask = self.ring_mask.to(CPU, )
        searching_edges = self.next[self.twins[self.v2e]].clone()
        for i in range(1, self.max_degree - 1):
            relevant_rows = self.v2e[:, i + 1] != -1
            searching_edge = searching_edges[relevant_rows][:, i - 1]
            ma = torch.eq(searching_edge[:, None], self.v2e[relevant_rows])
            ma *= self.ring_mask[relevant_rows]
            ma = ma.bool()
            where = torch.where(ma)[1]
            swap()

    def __init__(self, raw_mesh: T_Mesh, sort=True):
        device = raw_mesh[0].device
        self.mesh = mesh_utils.to(raw_mesh, CPU)
        self.vs_degree = self.init_v_degree(raw_mesh)
        self.max_degree = self.vs_degree.max().item()
        self.v2e, self.e2f, self.edges, self.twins, self.next, self.unique_edges_id = self.build_ds()
        self.ring_mask = torch.ne(self.v2e, -1)
        self.prev = self.next[self.next]
        if self.is_manifold and self.is_closed and sort:
            self.sort_edges()
        self.v2f_: TN = None
        self.areas_normals_ = None
        self.to(device)

#
# # todo merge with meshDS
# class NonManifoldDS(GeometryDS):
#
#     def to(self, device: D) -> NonManifoldDS:
#         self.mesh = mesh_utils.to(self.mesh, device)
#         self.ring_mask, self.vs_degree = mesh_utils.to((self.ring_mask, self.vs_degree), device)
#         self.v2e, self.e2f, self.edges = mesh_utils.to((self.v2e, self.e2f, self.edges), device)
#         self.twins, self.next = mesh_utils.to((self.twins, self.next), device)
#         self.unique_edges_id, self.v2f_, self.areas_normals_ = mesh_utils.to((self.unique_edges_id, self.v2f_,
#                                                                               self.areas_normals_), device)
#         return self
#
#     @property
#     def vs_normals(self):
#         pass
#
#     def __init__(self, raw_mesh: T_Mesh):
#         device = raw_mesh[0].device
#         self.mesh = mesh_utils.to(raw_mesh, CPU)
#         self.vs_degree = MeshDS.init_v_degree(raw_mesh)
#         self.max_degree = self.vs_degree.max().item()
#         self.areas_normals_ = None
#         self.to(device)
#         pass


class MeshDSCircle(MeshDS):

    def circulate(self) -> TS:
        num_circles, circle_size = self.v2e.shape
        circles = torch.arange(num_circles)
        replace = self.ring_mask.sum(1).long() - 1
        base_inds = torch.arange(circle_size).unsqueeze(1)
        base_inds = torch.cat(((base_inds - 1) % circle_size, base_inds, (base_inds + 1) % circle_size), dim=1)
        base_inds = base_inds.repeat(num_circles, 1, 1)
        base_inds[:, 0, 0] = replace
        base_inds[circles, replace, -1] = 0
        base_inds[:, :, :] += circle_size * circles[:, None, None]
        reduced_rings = base_inds[:, :, :-1]
        base_inds = base_inds.flatten()
        return base_inds, reduced_rings.flatten()

    def __init__(self, raw_mesh: T_Mesh):
        super(MeshDSCircle, self).__init__(raw_mesh)
        self.circular_emb_idx, self.circular_idx = self.circulate()
        self.to(self.device)




class MeshChecker:

    def check_non_manifold(self) -> T:

        def insert_face():
            new_edges = face[edge_extract]
            new_edges_id = face_id * 3 + zero_one_two
            edges[new_edges_id] = new_edges
            new_edges = new_edges.sort(1)[0]
            for i in range(3):
                edge = (new_edges[i, 0].item(), new_edges[i, 1].item())
                if edge not in edges_degree:
                    # edge2key[edge] = new_edges_id[i]
                    edges_degree[edge] = 0
                edges_degree[edge] += 1

        edges_degree = dict()
        edges = torch.zeros(self.faces.shape[0] * 3, 2, dtype=torch.int64)
        zero_one_two = torch.arange(3)
        # one_two_zero = (zero_one_two + 1) % 3
        edge_extract = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int64)
        # edge2key = dict()
        for face_id, face in enumerate(self.faces):
            insert_face()
        non_manifold_edges = list(map(lambda edge: list(edge), filter(lambda key: edges_degree[key] > 2, edges_degree)))
        non_manifold_edges = torch.tensor(non_manifold_edges, dtype=torch.int64)
        return non_manifold_edges

    @property
    def non_manifold_vs(self) -> TS:
        return self.non_manifold_edges.flatten().unique()

    @property
    def num_manifold(self) -> int:
        return self.non_manifold_edges.shape[0]

    @property
    def sanity(self):
        return self.faces.max() < self.vs.shape[0]

    def __call__(self) -> bool:
        return reduce(lambda a, b: a and b, [self.sanity, self.num_manifold == 0])

    def __len__(self):
        return self.vs.shape[0]

    @property
    def vs(self):
        return self.mesh[0]

    @property
    def faces(self):
        return self.mesh[1]

    def __init__(self, raw_mesh: T_Mesh):
        self.mesh = raw_mesh
        self.non_manifold_edges = self.check_non_manifold()

