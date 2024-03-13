from __future__ import annotations
import abc
from custom_types import *
import vtk
from utils import files_utils
from ui_sketch import ui_utils, ui_controllers, inference_processing, gaussian_status, sketch_inference
import options
import vtk.util.numpy_support as numpy_support
import constants
from data_loaders import augment_clipcenter
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import igl
from PIL import Image



def to_local(func):
    def inner(self: TransitionController, mouse_pos: Optional[Tuple[int, int]], *args, **kwargs):
        if mouse_pos is not None:
            size_full = self.render.GetRenderWindow().GetScreenSize()
            left, bottom, right, top = viewport = self.render.GetViewport()
            size = size_full[0] * (right - left), size_full[1] * (top - bottom)
            mouse_pos = float(mouse_pos[0]) / size[0] - .5, float(mouse_pos[1]) / size[1] - .5
            mouse_pos = torch.tensor([mouse_pos[0], mouse_pos[1]])
        return func(self, mouse_pos, *args, **kwargs)
    return inner


class TransitionController:

    @property
    def moving_axis(self) -> int:
        return {ui_utils.EditDirection.X_Axis: 0,
                ui_utils.EditDirection.Y_Axis: 2,
                ui_utils.EditDirection.Z_Axis: 1}[self.edit_direction]

    def get_delta_translation(self, mouse_pos: T) -> ARRAY:
        delta_3d = np.zeros(3)
        axis = self.moving_axis
        vec = mouse_pos - self.origin_mouse
        delta = torch.einsum('d,d', vec, self.dir_2d[:, axis])
        delta_3d[axis] = delta * self.camera.GetDistance()
        return delta_3d

    def get_delta_rotation(self, mouse_pos: T) -> ARRAY:
        projections = []
        for pos in (self.origin_mouse, mouse_pos):
            vec = pos - self.transition_origin_2d
            projection = torch.einsum('d,da->a', vec, self.dir_2d)
            projection[self.moving_axis] = 0
            projection = nnf.normalize(projection, p=2, dim=0)
            projections.append(projection)
        sign = (projections[0][(self.moving_axis + 2) % 3] * projections[1][(self.moving_axis + 1) % 3]
                - projections[0][(self.moving_axis + 1) % 3] * projections[1][(self.moving_axis + 2) % 3]).sign()
        angle = (torch.acos(torch.einsum('d,d', *projections)) * sign).item()
        return ui_utils.get_rotation_matrix(angle, self.moving_axis)

    def get_delta_scaling(self, mouse_pos: T) -> ARRAY:
        raise NotImplementedError

    def toggle_edit_direction(self, direction: ui_utils.EditDirection):
        self.edit_direction = direction

    @to_local
    def get_transition(self, mouse_pos: Optional[T]) -> ui_utils.Transition:
        transition = ui_utils.Transition(self.transition_origin.numpy(), self.transition_type)
        if mouse_pos is not None:
            if self.transition_type is ui_utils.EditType.Translating:
                transition.translation = self.get_delta_translation(mouse_pos)
            elif self.transition_type is ui_utils.EditType.Rotating:
                transition.rotation = self.get_delta_rotation(mouse_pos)
            elif self.transition_type is ui_utils.EditType.Scaling:
                transition.rotation = self.get_delta_scaling(mouse_pos)
        return transition

    @to_local
    def init_transition(self, mouse_pos: Tuple[int, int], transition_origin: T, transition_type: ui_utils.EditType):
        transform_mat_vtk = self.camera.GetViewTransformMatrix()
        dir_2d = torch.zeros(3, 4)
        for i in range(3):
            for j in range(4):
                dir_2d[i, j] = transform_mat_vtk.GetElement(i, j)
        self.transition_origin = transition_origin
        transition_origin = torch.tensor(transition_origin.tolist() + [1])
        transition_origin_2d = torch.einsum('ab,b->a', dir_2d, transition_origin)
        self.transition_origin_2d = transition_origin_2d[:2] / transition_origin_2d[-1].abs()
        # print(f"<{self.transition_origin[0]}, {self.transition_origin[1]}>")
        # print(mouse_pos)
        self.origin_mouse, self.dir_2d = mouse_pos, nnf.normalize(dir_2d[:2, :3], p=2, dim=1)
        self.transition_type = transition_type

    @property
    def camera(self):
        return self.render.GetActiveCamera()

    def __init__(self, render: ui_utils.CanvasRender):
        self.render = render
        self.transition_origin = torch.zeros(3)
        self.transition_origin_2d = torch.zeros(2)
        self.origin_mouse, self.dir_2d = torch.zeros(2), torch.zeros(2, 3)
        self.edit_direction = ui_utils.EditDirection.X_Axis
        self.transition_type = ui_utils.EditType.Translating


class VtkTimerCallback:

    def __init__(self, steps, iren, camera, callback):
        self.scrolling = False
        self.timer_count = 0
        self.steps = steps
        self.iren = iren
        self.camera = camera
        self.pos_x = camera.GetPosition()[0]
        self.callback = callback
        self.timer_id = None

    @staticmethod
    def ease_in_out(alpha: float):
        if alpha < .5:
            return 2 * alpha ** 2
        else:
            return 1 - ((-2 * alpha + 2) ** 4) / 2

    @staticmethod
    def ease_in(alpha: float):
        return alpha ** 2

    @staticmethod
    def ease_out(alpha: float):
        if alpha == 1:
            return 1
        return 1 - 2 ** (-4 * alpha) / (1 - 2 ** (-4))

    def init_scrolling(self):
        if self.timer_id is not None:
            self.iren.DestroyTimer(self.timer_id)
            self.timer_id = None
        self.scrolling = True
        self.timer_count = 0
        self.timer_id = None

    def execute(self, obj, event):
        alpha = self.ease_out(float(self.timer_count + 1) / self.steps)
        # alpha = float(self.timer_count + 1) / self.steps
        pos_x = self.pos_x + 4 * alpha
        self.camera.SetPosition(pos_x, 0, 1)
        self.camera.SetFocalPoint(pos_x, 0, 0)
        iren = obj
        iren.GetRenderWindow().Render()
        self.timer_count += 1
        if self.timer_count == self.steps - 1:
            self.pos_x = self.pos_x + 4
            self.scrolling = False
            self.callback()


class StagedCanvas(ui_utils.CanvasRender, abc.ABC):

    def reset(self):
        self.stage.reset()

    def vote(self, *actors: Optional[vtk.vtkActor]):
        self.stage.vote(*actors)

    @abc.abstractmethod
    def after_draw(self, changed: List[int], select: bool) -> bool:
        raise NotImplementedError

    def aggregate_votes(self, select: bool) -> bool:
        changed = self.stage.aggregate_votes()
        return self.after_draw(changed, select)

    def save(self, root: str):
        self.stage.save(root, ui_controllers.filter_by_selection)

    def __init__(self, opt, viewport:Tuple[float, float, float, float], render_window: vtk.vtkRenderWindow,
                 bg_color: ui_utils.RGB_COLOR, stroke_color: Optional[ui_utils.RGBA_COLOR]):
        super(StagedCanvas, self).__init__(viewport, render_window, bg_color, stroke_color)
        self.stage = ui_controllers.GmmMeshStage(opt, -1, self, -1, ui_utils.ViewStyle((255, 255, 255),
                                                                                      (255, 255, 255), ui_utils.bg_target_color, 1))


class RenderSketch(ui_utils.CanvasRender):

    def aggregate_votes(self, select: bool) -> bool:
        return False

    def vote(self, *actors: Optional[vtk.vtkActor]):
        return

    def set_brush(self, is_draw: bool):
        super(RenderSketch, self).set_brush(is_draw)
        if is_draw:
            self.set_stroke_width(self.stroke_draw)
        else:
            self.set_stroke_width(10)

    def save_image(self):
        image = self.get_sketch()
        files_utils.save_image(image[:, :, :3], "./assets/tmp/tmp_sketch.png")
        return
    
    def add_image(self, filename="tmp"):
        image = files_utils.load_image(f"{constants.PROJECT_ROOT}/{filename}")
        print(image.shape)
        self.set_sketch(image[:, :, 0])


    def get_sketch(self):
        image_data = self.canvas.GetOutput().GetPointData().GetScalars()
        image_data = numpy_support.vtk_to_numpy(image_data)
        dimension = self.canvas.GetOutput().GetExtent()
        image = image_data.reshape((dimension[3] + 1, dimension[1] + 1, 4))
        image = image[::-1]
        image = image[:, :, :3]
        try:
            image = augment_clipcenter.augment_cropped_square(image)
        except:
            print("Crop failed. Returning the image without cropping. Is the image empty?")
        return image.copy()

    def fill_bg(self):
        self.canvas.SetDrawColor(255, 255, 255, 0)
        self.canvas.FillBox(0, self.width, 0, self.height)
        self.canvas.SetDrawColor(*self.stroke_color)
        self.canvas.Update()

    def erase_all(self):
        self.clear()
        self.fill_bg()

    def reset(self):
        self.erase_all()

    def clear(self):
        self.last_point = None
    
    def set_sketch(self, sketch: ARRAY):
        self.canvas.SetDrawColor(255, 255, 255, 0)
        self.canvas.FillBox(0, self.width, 0, self.height)
        self.canvas.SetDrawColor(*self.stroke_color)
        dimension = self.canvas.GetOutput().GetExtent()
        _, height, _, width, _, _ = dimension
        height_sketch, width_sketch = sketch.shape
        new_sketch = np.full((height, width), 255, dtype=np.uint8)
        start_w = (width - width_sketch)//2
        end_w = start_w+width_sketch
        start_h = (height - height_sketch)//2
        end_h = start_h+height_sketch
        new_sketch[start_w:end_w, start_h:end_h] = sketch

        for i in range(width):
            for j in range(height):
                if (new_sketch[i, j] != 255):
                    self.canvas.FillBox(j, j, width-1-i, width-i-1)
        self.canvas.Update()

    def __init__(self, viewport:Tuple[float, float, float, float], render_window: vtk.vtkRenderWindow, iren,
                 bg_color: ui_utils.RGB_COLOR, stroke_color: ui_utils.RGBA_COLOR):
        self.stroke_draw = 1
        super(RenderSketch, self).__init__(viewport, render_window, bg_color, stroke_color)
        self.render_window = render_window
        self.source_stage: Optional[ui_controllers.GmmMeshStage] = None
        self.GetActiveCamera().SetPosition(0, 0, 4)
        
        self.set_stroke_width(self.stroke_draw)


class RenderMain(StagedCanvas):

    def toggle_edit_direction(self, direction: ui_utils.EditDirection):
        self.transition_controller.toggle_edit_direction(direction)

    def clear_selection(self) -> bool:
        is_changed = False
        for gaussian in self.selected_gaussians:
            gaussian.toggle_selection()
            is_changed = True
        return is_changed

    @property
    def selected_gaussians(self) -> Iterable[gaussian_status.GaussianStatus]:
        return filter(lambda x: x.is_selected, self.stage.gmm)

    def temporary_transition(self, mouse_pos: Optional[Tuple[int, int]] = None, end=False) -> bool:
        transition = self.transition_controller.get_transition(mouse_pos)
        is_change = False
        for gaussian in self.selected_gaussians:
            if end:
                is_change = gaussian.end_transition(transition) or is_change
            else:
                is_change = gaussian.temporary_transition(transition) or is_change
        return is_change

    def end_transition(self, mouse_pos: Optional[Tuple[int, int]]) -> bool:
        return self.temporary_transition(mouse_pos, True)

    def init_transition(self, mouse_pos, transition_type: ui_utils.EditType):
        center = list(map(lambda x: x.mu_baked, self.selected_gaussians))
        if len(center) == 0:
            return
        center = torch.from_numpy(np.stack(center, axis=0).mean(0))
        self.transition_controller.init_transition(mouse_pos, center, transition_type)

    def reset(self):
        self.stage.remove_all()
        self.stage_mapper = {}

    def make_twins(self, toggled: List[gaussian_status.GaussianStatus], new_addresses: List[str]):
        if len(new_addresses) == 2:
            self.stage.make_twins(*new_addresses)
        else:
            if toggled[0].twin is not None and toggled[0].twin.get_address() in self.stage_mapper:
                self.stage.make_twins(new_addresses[0], self.stage_mapper[toggled[0].twin.get_address()])

    def update_gmm(self, stage: ui_controllers.GmmMeshStage, changed: List[int]):
        for item in changed:
            is_toggled, toggled = stage.toggle_inclusion_by_id(item, self.is_draw)
            if is_toggled:
                if toggled[0].included:
                    new_addresses = self.stage.add_gaussians(toggled)
                    for gaussian, new_address in zip(toggled, new_addresses):
                        self.stage_mapper[gaussian.get_address()] = new_address
                    self.make_twins(toggled, new_addresses)
                else:
                    addresses = [gaussian.get_address() for gaussian in toggled]
                    addresses = list(filter(lambda x: x in self.stage_mapper, addresses))
                    self.stage.remove_gaussians([self.stage_mapper[address] for address in addresses])
                    for address in addresses:
                        del self.stage_mapper[address]

    def update_mesh(self, res=128):
        if self.model_process is not None:
            self.model_process.get_mesh(res)
            return True
        return False

    def after_draw(self, changed: List[int], select: bool) -> bool:
        changed = list(filter(lambda x: not self.stage.gmm[x].disabled and self.stage.gmm[x].is_selected != select, changed))
        for item in changed:
            self.stage.gmm[item].toggle_selection()
        return False

    def replace_mesh(self):
        self.reset()
        if self.model_process is not None:
            self.model_process.replace_mesh()

    def exit(self):
        if self.model_process is not None:
            self.model_process.exit()

    def save_info(self, task: str):
        selected = [not g.disabled and g.is_selected for i, g in enumerate(self.stage.gmm)]
        files_utils.save_pickle({'task': task, 'info': selected}, './assets/tmp/info')
        return selected

    def rebuild(self):
        if self.model_process is not None:
            selected = self.save_info('rebuild')
            if len(selected) > 0:
                self.model_process.get_mesh()
        return
    
    def rebuild_selected_fromsketch(self, sketch: ARRAY):
        self.last_sketch = sketch
        if self.model_process is not None:
            selected = self.save_info('rebuild_sketch')
            if len(selected) > 0:
                self.model_process.get_mesh()
        return

    def get_mesh(self, sketch: ARRAY):
        self.last_sketch = sketch
        if self.model_process is not None:
            _ = self.save_info('from_scratch')
            self.model_process.get_mesh()
    
    # a function that gets the mesh for real
    def get_mesh_real(self):
        mesh = self.model_process.get_mesh_real()
        return mesh

    def request_sketch(self):
        return self.last_sketch

    def __init__(self, opt, viewport:Tuple[float, float, float, float], render_window: vtk.vtkRenderWindow,
                 bg_color: ui_utils.RGB_COLOR, stroke_color: Optional[ui_utils.RGBA_COLOR], with_model: bool):
        super(RenderMain, self).__init__(opt, viewport, render_window, bg_color, stroke_color)
        if with_model:
            self.model_process = inference_processing.InferenceProcess(opt, self.stage.replace_mesh,
                                                                       self.request_sketch)
        else:
            self.model_process = None
        self.stage_mapper: Dict[str, str] = {}
        self.transition_controller = TransitionController(self)
        self.GetActiveCamera().SetPosition(3, 1, -3)
        self.last_sketch: Optional[ARRAY] = None

def render_mesh(mesh, np_camera_pos, np_matrix_proj, np_model_view_matrix, scaling = 0.15):
    V, F = mesh
    V, F = igl.loop(V, F)

    new_np_matrix_proj = np_matrix_proj.copy()
    new_np_matrix_proj[3, 2]= new_np_matrix_proj[3, 2]*scaling
    MVP  = new_np_matrix_proj @ np_model_view_matrix
    V = np.c_[V, np.ones(len(V))] @ MVP.T
    V /= V[:,3].reshape(-1,1)
    V = V[F]
    T = V[:,:,:2]
    Z = -V[:,:,2].mean(axis=1)
    zmin, zmax = Z.min(), Z.max()
    Z = (Z-zmin)/(zmax-zmin)
    C = plt.get_cmap("viridis")(Z)
    C = plt.get_cmap("gray")(Z)
    I = np.argsort(Z)
    T, C = T[I,:], C[I,:]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1],    
                    aspect=1, frameon=False)

    collection = PolyCollection(T, closed=True, linewidth=0.1,
                                facecolor=C, edgecolor="face")
    ax.add_collection(collection)

    # To remove the huge white borders
    ax.axis('off')
    ax.margins(0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    del fig
    del ax
    plt.close()
    return image_from_plot

def no_border(img):
    return not ((img[0, :, :] == np.uint8(255)).all() and (img[:, 0, :] == np.uint8(255)).all() and (img[img.shape[0]-1, :, :] == np.uint8(255)).all() and (img[:, img.shape[1]-1, :] == np.uint8(255)).all())

def render_mesh_optiscale(mesh, np_camera_pos, np_matrix_proj, np_model_view_matrix):
    scaling = 0.15
    render = render_mesh(mesh, np_camera_pos, np_matrix_proj, np_model_view_matrix, scaling)
    render_bad = no_border(render)
    scaling_low = 0.15
    scaling_high = 0.3
    if (render_bad):
        while (render_bad):
            scaling_low = scaling
            scaling *= 2
            render = render_mesh(mesh, np_camera_pos, np_matrix_proj, np_model_view_matrix, scaling)
            render_bad = no_border(render)
            scaling_high = scaling
        for i in range(5):
            # optimize here
            scaling = ((scaling_high - scaling_low) / 2) + scaling_low
            render = render_mesh(mesh, np_camera_pos, np_matrix_proj, np_model_view_matrix, scaling)
            render_bad = no_border(render)
            if (render_bad):
                scaling_low = scaling
            else:
                scaling_high = scaling
    
        render_bad = no_border(render)
        while (render_bad):
            scaling = ((scaling_high - scaling_low) / 2) + scaling_low
            render = render_mesh(mesh, np_camera_pos, np_matrix_proj, np_model_view_matrix, scaling)
            render_bad = no_border(render)
            scaling_low = scaling
    return render


def contour_render(img : ARRAY):
    img_blur = cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0) 
    edges = cv2.Canny(image=img_blur, threshold1=20, threshold2=120) 
    edges = 255 - edges
    return edges


class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    class MouseStatus:

        def update(self, pos: Tuple[int, int], selected_view: int) -> bool:
            is_changed = selected_view != self.selected_view or (pos[0] - self.last_pos[0]) ** 2 > 4 and (
                        pos[1] - self.last_pos[1]) ** 2 > 4
            self.selected_view = selected_view
            self.last_pos = pos
            return is_changed

        def __init__(self):
            self.last_pos = (0, 0)
            self.selected_view = 0

    def on_key_press(self, obj, event):
        key = self.interactor.GetKeySym()
        if type(key) is not str:
            return
        key: str = key.lower()
        if self.pondering:
            if key in ('g', 'r'):
                self.edit_status = {'g': ui_utils.EditType.Translating,
                                    'r': ui_utils.EditType.Rotating,
                                    's': ui_utils.EditType.Scaling}[key]
                click_pos = self.interactor.GetEventPosition()
                self.render_main.init_transition(click_pos, self.edit_status)
            elif key == 'escape':
                if self.render_main.clear_selection():
                    self.interactor.Render()
        elif self.in_transition:
            if key == 'escape':
                self.render_main.temporary_transition()
                self.interactor.Render()
                self.edit_status = ui_utils.EditType.Pondering
            elif key in ('x', 'y', 'z'):
                self.render_main.toggle_edit_direction({'x': ui_utils.EditDirection.X_Axis,
                                                   'y': ui_utils.EditDirection.Y_Axis,
                                                   'z': ui_utils.EditDirection.Z_Axis}[key])
        if key == 'return' or key == 'kp_enter':
            mesh = self.render_main.get_mesh_real()
            print("mesh", mesh)
            files_utils.export_mesh(mesh, "./assets/tmp/tmp_mesh")
        if key == 'control_l':
            self.render_main.replace_mesh()
            self.interactor.Render()

        return

    @property
    def interactor(self):
        return self.GetInteractor()

    def replace_render(self):
        self.OnLeftButtonDown()
        self.OnLeftButtonUp()
        return self.GetCurrentRenderer()

    def left_button_press_event(self, obj, event):
        cur_render = self.replace_render()
        if self.in_transition:
            click_pos = self.interactor.GetEventPosition()
            if self.render_main.end_transition(click_pos):
                self.render_main.update_mesh()
            self.edit_status = ui_utils.EditType.Pondering
        else:
            super(InteractorStyle, self).OnLeftButtonDown()

    def left_button_release_event(self, obj, event):
        super(InteractorStyle, self).OnLeftButtonUp()

    def right_button_release_event(self, obj, event):
        if self.marking:
            self.draw_end()
            self.edit_status = ui_utils.EditType.Pondering

    def update_default(self):
        click_pos = self.interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        cur_render = self.GetCurrentRenderer()
        if cur_render != self.selected_view:
            self.selected_view = cur_render
        elif cur_render is None:
            return None, False, click_pos
        picker.Pick(click_pos[0], click_pos[1], 0, cur_render)
        return picker, self.mouse_status.update((click_pos[0], click_pos[1]), cur_render), click_pos

    def draw_end(self):
        self.draw_view.clear()
        if self.draw_view.aggregate_votes(self.select_mode):
            self.render_main.update_mesh()
        self.interactor.Render()
        #TODO: Here, save mesh
        self.render_sketch.save_image()
        

    def right_button_press_event(self, obj, event):
        self.OnRightButtonDown()
        self.OnRightButtonUp()
        _ = self.update_default()
        if self.pondering:
            self.draw_view = self.selected_view
            self.edit_status = ui_utils.EditType.Marking
        return

    def on_mouse_wheel_backward(self, obj, event):
        self.OnMouseWheelBackward()

    def on_mouse_wheel_forward(self, obj, event):
        super(InteractorStyle, self).OnMouseWheelForward()

    def middle_button_press_event(self, obj, event):
        super(InteractorStyle, self).OnMiddleButtonDown()

    def middle_button_release_event(self, obj, event):
        super(InteractorStyle, self).OnMiddleButtonUp()

    def get_trace(self):
        picker, is_changed, pos_2d = self.update_default()
        if picker is not None:
            points = self.draw_view.draw(pos_2d)
            actors = []
            for point in points:
                picker.Pick(point[0], point[1], 0, self.draw_view)
                actors.append(picker.GetActor())
            self.draw_view.vote(*actors)
            self.interactor.Render()
        return 0

    def on_mouse_move(self, obj, event):
        if self.marking:
            self.get_trace()
        elif self.in_transition:
            click_pos = self.interactor.GetEventPosition()
            if self.render_main.temporary_transition(click_pos):
                self.interactor.Render()
        else:
            super(InteractorStyle, self).OnMouseMove()

    def add_buttons(self, interactor):

        def toggle_select_mode(button, __):
            self.select_mode = button.GetRepresentation().GetState() == 0
            self.render_sketch.set_brush(self.select_mode)
            self.render_main.set_brush(self.select_mode)

        def reset(_, __):
            self.render_sketch.reset()
            self.render_main.reset()
            self.interactor.Render()

        def rebuild(_, __):
            self.render_main.rebuild()
            return

        def build(_, __):
            if not self.rendered_sketch:
                sketch = self.render_sketch.get_sketch()
                self.render_main.get_mesh(sketch)
                self.OnMouseWheelBackward()
        
        def build_selected(_, __):
            """Rebuild only the selected part"""
            if not self.rendered_sketch:
                sketch = self.render_sketch.get_sketch()
                self.render_main.rebuild_selected_fromsketch(sketch)
        
        def contour_drawing(_, __):
            display_steps = False
            mesh = self.render_main.get_mesh_real()

            def get_numpy_4x4(vtkmatrix4x4):
                result = np.zeros((4,4))
                for i in range(4):
                    for j in range(4):
                        result[i,j] = vtkmatrix4x4.GetElement(i, j)
                return result

            vtk_camera = self.render_main.GetActiveCamera()
            vtk_matrix_proj = vtk_camera.GetProjectionTransformMatrix(self.render_main.canvas_render)
            np_matrix_proj = get_numpy_4x4(vtk_matrix_proj)

            # Manual bug fixing here...
            if abs(np_matrix_proj[2,2] + 1.00002) < 1e-4 and abs(np_matrix_proj[3, 2]+0.0200002) < 1e-4:
                np_matrix_proj[2,2] = -2.32557446 
                np_matrix_proj[3,2] = -8.68914846


            vtk_model_view_matrix = vtk_camera.GetModelViewTransformMatrix()
            np_model_view_matrix = get_numpy_4x4(vtk_model_view_matrix)


            camera_pos = vtk_camera.GetPosition()
            np_camera_pos = np.array(camera_pos)


            rendered_image = render_mesh_optiscale(mesh, np_camera_pos, np_matrix_proj, np_model_view_matrix)
            plt.close()

            tmp_depthmap = Image.fromarray(rendered_image)
            tmp_depthmap.save("./assets/tmp/tmp_depthmap.png")

            edges = contour_render(rendered_image)
            tmp_edges = Image.fromarray(edges)
            tmp_edges.save("./assets/tmp/tmp_edges.png")
            if (display_steps):
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(rendered_image)
                axs[0].axis("off")
                axs[1].imshow(edges)
                axs[1].axis("off")
                plt.figure(fig)
                plt.axis("off")
                plt.show()
            try:
                edges = np.expand_dims(edges, axis=2)
                edges = np.concatenate((edges, edges, edges), axis=2)
                edges = augment_clipcenter.augment_cropped_square(edges)
                if (display_steps):
                    plt.axis('off')
                    plt.imshow(edges)
                    plt.show()
                edges = edges[:, :, 0]
            except Exception as e:
                print(e)
                pass

            self.render_sketch.set_sketch(edges)


        button_pencil = ui_utils.ImageButton([f"{constants.UI_RESOURCES}icons-03.png", f"{constants.UI_RESOURCES}icons-04.png"],
                                             interactor, self.render_sketch, (.09, .09), (0.01, .98), toggle_select_mode, full_size=(1., 1.))
        button_reset = ui_utils.ImageButton([f"{constants.UI_RESOURCES}icons-16.png"], interactor, self.render_main,
                                            (.1, .1), (0.01, .98), reset, full_size=(1., 1.))

        button_rebuild = ui_utils.ImageButton([f"{constants.UI_RESOURCES}icons-15.png"], interactor, self.render_main,
                                              (.09, .09), (0.07, 0.97), rebuild, full_size=(1., 1.))

        button_spaghetti = ui_utils.ImageButton([f"{constants.UI_RESOURCES}icons-17.png"], interactor, self.render_sketch,
                                              (.15, .15), (0.01, 0.05), build, full_size=(1., 1.))
        
        button_redraw = ui_utils.ImageButton([f"{constants.UI_RESOURCES}icons-19.png"], interactor, self.render_sketch,
                                              (.15, .15), (0.09, 0.1), build_selected, full_size=(1., 1.))

        button_contour = ui_utils.ImageButton([f"{constants.UI_RESOURCES}icons-20.png"], interactor, self.render_sketch,
                                              (.15, .15), (0.13, 0.1), contour_drawing, full_size=(1., 1.))

        return button_pencil, button_reset, button_rebuild, button_spaghetti, button_redraw, button_contour

    @property
    def marking(self) -> bool:
        return self.edit_status == ui_utils.EditType.Marking

    @property
    def pondering(self) -> bool:
        return self.edit_status == ui_utils.EditType.Pondering

    @property
    def translating(self) -> bool:
        return self.edit_status == ui_utils.EditType.Translating

    @property
    def rotating(self) -> bool:
        return self.edit_status == ui_utils.EditType.Rotating

    @property
    def scaling(self) -> bool:
        return self.edit_status == ui_utils.EditType.Scaling

    @property
    def in_transition(self):
        return self.translating or self.rotating or self.scaling

    def init_observers(self):
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.left_button_press_event)
        self.AddObserver(vtk.vtkCommand.LeftButtonReleaseEvent, self.left_button_release_event)
        self.AddObserver(vtk.vtkCommand.RightButtonReleaseEvent, self.right_button_release_event)
        self.AddObserver(vtk.vtkCommand.RightButtonPressEvent, self.right_button_press_event)
        self.AddObserver(vtk.vtkCommand.MouseWheelBackwardEvent, self.on_mouse_wheel_backward)
        self.AddObserver(vtk.vtkCommand.MouseWheelForwardEvent, self.on_mouse_wheel_forward)
        self.AddObserver(vtk.vtkCommand.MiddleButtonPressEvent, self.middle_button_press_event)
        self.AddObserver(vtk.vtkCommand.MiddleButtonReleaseEvent, self.middle_button_release_event)
        self.AddObserver(vtk.vtkCommand.MouseMoveEvent, self.on_mouse_move)
        self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.on_key_press)
        self.AddObserver(vtk.vtkCommand.CharEvent, lambda _, __: None)

    def __init__(self, opt: options.Options, render_sketch: RenderSketch, render_main: RenderMain, interactor):
        super(InteractorStyle, self).__init__()
        self.mouse_status = self.MouseStatus()
        self.render_main = render_main
        self.edit_status = ui_utils.EditType.Pondering
        self.edit_direction = ui_utils.EditDirection.X_Axis
        self.select_mode = True
        self.selected_view: Optional[StagedCanvas] = None
        self.draw_view: Optional[StagedCanvas] = None
        self.render_sketch = render_sketch
        self.sketch_buffer = None
        self.init_observers()
        self.buttons = self.add_buttons(interactor)
        self.rendered_sketch = False


def main(with_model: bool = True):
    opt = options.SketchOptions(tag = "chairs", spaghetti_tag="chairs_sym_hard")

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 600)
    render_window.SetNumberOfLayers(3)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    renderer_main = RenderMain(opt, (0., 0., .5, 1.), render_window, ui_utils.bg_source_color,
                                          list(ui_utils.bg_target_color) + [200], with_model)
    renderer_sketch = RenderSketch((0.5, 0., 1., 1.), render_window, interactor, [255, 255, 255],
                                   [0, 0, 0] + [255])

    render_window.Render()
    interactor.Initialize()
    style = InteractorStyle(opt, renderer_sketch, renderer_main, interactor)
    interactor.SetInteractorStyle(style)
    renderer_sketch.fill_bg()
    render_window.Render()
    interactor.Start()
    del interactor
    del render_window
    renderer_main.exit()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    main(True)
        
