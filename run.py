from custom_types import *
from constants import ASSETS_ROOT
import options
from utils import files_utils
from ui_sketch import sketch_inference
from data_loaders import augment_clipcenter


def get_mesh(spaghetti, zh):
    out_z, gmms = spaghetti.model.occ_former.forward_mid(zh.unsqueeze(0).unsqueeze(0))
    out_z = spaghetti.model.merge_zh_step_a(out_z, gmms)
    out_z, _ = spaghetti.model.affine_transformer.forward_with_attention(out_z[0][:].unsqueeze(0))
    mesh = spaghetti.get_mesh(out_z[0], 256, None)
    if mesh is None:
        return None, None
    gmm = gmms
    return gmm[0], mesh

def sketch2mesh(path: str, model):
    sketch = files_utils.load_image(path)
    sketch = augment_clipcenter.augment_cropped_square(sketch, 256)
    gmm, mesh, zh_0 = model.sketch2mesh(sketch, get_zh=True)
    return gmm, mesh, zh_0, sketch

def main(input: str, to_save = True):
    opt = options.SketchOptions(tag = "chairs", spaghetti_tag="chairs_sym_hard")
    model = sketch_inference.SketchInference(opt)
    gmm, mesh, zh_0, sketch = sketch2mesh(input, model)

    folder = f"{ASSETS_ROOT}/output/"
    if to_save:
        filename_gmm = f'{folder}/gmm'
        filename_mesh = f'{folder}/mesh_res'
        filename_zh_0 = f'{folder}/zh_0'
        filename_input_sketch = f'{folder}/cropped_sketch.png'
        files_utils.export_gmm(gmm, 0, filename_gmm)
        files_utils.export_mesh(mesh, filename_mesh)
        files_utils.save_image(sketch, filename_input_sketch)
        files_utils.save_np(zh_0, filename_zh_0)
        print("Done - Saved files to", folder)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--to_save', type=bool, default=True)
    args = parser.parse_args()
    main(args.input, args.to_save)