import os
from PIL import Image
from custom_types import *
import multiprocessing as mp
from multiprocessing import synchronize
import options
import constants
from ui_sketch.sketch_inference import SketchInference
from utils import files_utils
import ctypes
from pynput.keyboard import Key, Controller


class UiStatus(Enum):
    Waiting = 0
    GetMesh = 1
    SetGMM = 2
    SetMesh = 3
    ReplaceMesh = 4
    Exit = 5


def value_eq(value: mp.Value, status: UiStatus) -> bool:
    return value.value == status.value


def value_neq(value: mp.Value, status: UiStatus) -> bool:
    return value.value != status.value


def set_value(value: mp.Value, status: UiStatus):
    with value.get_lock():
        value.value = status.value
    print({0: 'Waiting', 1: 'GetMesh', 2: 'SetGMM', 3: 'SetMesh', 4: 'ReplaceMesh', 5: 'Exit'}[status.value])


def set_value_if_eq(value: mp.Value, status: UiStatus, check: UiStatus):
    if value_eq(value, check):
        set_value(value, status)


def set_value_if_neq(value: mp.Value, status: UiStatus, check: UiStatus):
    if value_neq(value, check):
        set_value(value, status)


def store_mesh(mesh: T_Mesh, shared_meta: mp.Array, shared_vs: mp.Array, shared_faces: mp.Array):

    def store_tensor(tensor: T, s_array: mp.Array, dtype, meta_index):
        nonlocal shared_meta_
        s_array_ = to_np_arr(s_array, dtype)
        array_ = tensor.detach().cpu().flatten().numpy()
        arr_size = array_.shape[0]
        s_array_[:array_.shape[0]] = array_
        shared_meta_[meta_index] = arr_size

    if mesh is not None:
        shared_meta_ = to_np_arr(shared_meta, np.int32)
        vs, faces = mesh
        store_tensor(vs, shared_vs, np.float32, 0)
        store_tensor(faces, shared_faces, np.int32, 1)


def store_sketch(shared_image: mp.Array, sketch: ARRAY):
    if sketch.shape[0] != 256 or sketch.shape[1] != 256:
        image = Image.fromarray(sketch)
        image = image.resize((256, 256), Image.BICUBIC)
        sketch = V(image)
    s_array_ = to_np_arr(shared_image, np.uint8)
    array_ = sketch.reshape(256 ** 2 * 3)
    s_array_[:] = array_
    return


def load_mesh(shared_meta: mp.Array, shared_vs: mp.Array, shared_faces: mp.Array) -> V_Mesh:

    def load_array(s_array: mp.Array, dtype, meta_index) -> ARRAY:
        nonlocal shared_meta_
        s_array_ = to_np_arr(s_array, dtype)
        array_ = s_array_[: shared_meta_[meta_index]].copy()
        array_ = array_.reshape((-1, 3))
        return array_

    shared_meta_ = to_np_arr(shared_meta, np.int32)
    vs = load_array(shared_vs, np.float32, 0)
    faces = load_array(shared_faces, np.int32, 1)
    return vs, faces


def store_zh(zh: T, shared_zh : mp.Array):
    def store_tensor(tensor: T, s_array: mp.Array, dtype):
        s_array_ = to_np_arr(s_array, dtype)
        array_ = tensor.detach().cpu().flatten().numpy()
        s_array_[:array_.shape[0]] = array_
    store_tensor(zh, shared_zh, np.float32)    

def load_zh(shared_zh : mp.Array):
    zh = to_np_arr(shared_arr=shared_zh, dtype = np.float32)
    zh = torch.from_numpy(zh)
    zh = zh.reshape((1, 16, -1))
    return zh


def load_sketch(shared_image: mp.Array) -> ARRAY:
    shared_arr = to_np_arr(shared_image, np.uint8)
    return shared_arr.reshape((256, 256, 3))


def inference_process(opt: options.SketchOptions, wake_condition: synchronize.Condition,
                      sleep__condition: synchronize.Condition, status: mp.Value, shared_image: mp.Array,
                      shared_meta: mp.Array, shared_vs: mp.Array, shared_faces: mp.Array, shared_zh : mp.Array):
    model = SketchInference(opt)
    keyboard = Controller()
    while value_neq(status, UiStatus.Exit):
        while value_eq(status, UiStatus.Waiting):
            with sleep__condition:
                sleep__condition.wait()
        if value_eq(status, UiStatus.GetMesh):
            set_value(status, UiStatus.SetGMM)
            set_value_if_eq(status, UiStatus.SetMesh, UiStatus.SetGMM)
            info = files_utils.load_pickle('./assets/tmp/info')
            task, select = info['task'], info['info']
            if task == 'from_scratch':
                sketch = load_sketch(shared_image)
                gmm, mesh, zh = model.sketch2mesh(sketch, get_zh=True)
                store_zh(zh, shared_zh)
            elif task == 'rebuild_sketch':
                sketch = load_sketch(shared_image)
                _zh = load_zh(shared_zh)
                gmm, mesh, zh = model.sketch2mesh_partial(sketch, select, _zh, get_zh=True)
                store_zh(zh, shared_zh)
            else:
                gmm, mesh = model.rebuild_mesh(select)
            if mesh is not None:
                files_utils.export_gmm(gmm, 0, "./assets/tmp/tmp_gmm")
                store_mesh(mesh, shared_meta, shared_vs, shared_faces)
                keyboard.press(Key.ctrl_l)
                keyboard.release(Key.ctrl_l)
            set_value_if_eq(status, UiStatus.ReplaceMesh, UiStatus.SetMesh)
    with wake_condition:
        wake_condition.notify_all()
    return 0


def to_np_arr(shared_arr: mp.Array, dtype) -> ARRAY:
    return np.frombuffer(shared_arr.get_obj(), dtype=dtype)


class InferenceProcess:

    skips = (2, 3, 9, 1, 3)

    def exit(self):
        set_value(self.status, UiStatus.Exit)
        with self.wake_condition:
            self.wake_condition.notify_all()
        self.model_process.join()

    def replace_mesh(self):
        mesh = load_mesh(self.shared_meta, self.shared_vs, self.shared_faces)
        self.fill_ui_mesh(mesh)
        set_value_if_eq(self.status, UiStatus.Waiting, UiStatus.ReplaceMesh)

    def get_mesh(self):
        if value_neq(self.status, UiStatus.SetGMM):
            sketch = self.request_sketch()
            store_sketch(self.shared_image, sketch)
            set_value_if_neq(self.status, UiStatus.GetMesh, UiStatus.GetMesh)
            with self.wake_condition:
                self.wake_condition.notify_all()
        return
    
    def get_mesh_real(self):
        mesh = load_mesh(self.shared_meta, self.shared_vs, self.shared_faces)
        return mesh

    def __init__(self, opt, fill_ui_mesh: Callable[[V_Mesh], None], request_sketch: Callable[[], ARRAY]):
        self.opt = opt
        self.status = mp.Value('i', UiStatus.Waiting.value)
        self.request_sketch = request_sketch
        self.sleep_condition = mp.Condition()
        self.wake_condition = mp.Condition()
        self.shared_vs = mp.Array(ctypes.c_float, constants.MAX_VS * 3)
        self.shared_image = mp.Array(ctypes.c_int8, 256 ** 2 * 3)
        self.shared_faces = mp.Array(ctypes.c_int, constants.MAX_VS * 8)
        self.shared_meta = mp.Array(ctypes.c_int, 2)
        self.shared_zh = mp.Array(ctypes.c_float, 16*512)
        self.model_process = mp.Process(target=inference_process,
                                        args=(opt, self.sleep_condition, self.wake_condition, self.status,
                                              self.shared_image, self.shared_meta, self.shared_vs,
                                              self.shared_faces, self.shared_zh))
        self.fill_ui_mesh = fill_ui_mesh
        self.model_process.start()