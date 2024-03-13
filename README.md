# SENS: Part-Aware Sketch-based Implicit Neural Shape Modeling

Official implementation of our paper, published at Eurographics 2024.

- DOI: 10.1111/cgf.15015
- arxiv: https://arxiv.org/abs/2306.06088
- website: 

## Pre-trained models

To run our method, you need to download two pre-trained models: one for the sketch-to-shape decoder network, and one for the shape decoder (SPAGHETTI) itself.
Download the .zip file at this address: [https://drive.google.com/file/d/15b98DQL3ebL_SaccRys0zJimCxppxaVo/view?usp=sharing](https://drive.google.com/file/d/15b98DQL3ebL_SaccRys0zJimCxppxaVo/view?usp=sharing)

Extract it and put the two folders 'occ_gmm_chairs_sym_hard' and 'sketch2spaghetti_chairs' in `assets/checkpoints`

## Data

The complete preprocessed chair dataset can be found here: [https://drive.google.com/file/d/10YHyS2O-SPwKFBR65aZE0KEpvaUQVfzA/view?usp=sharing](https://drive.google.com/file/d/10YHyS2O-SPwKFBR65aZE0KEpvaUQVfzA/view?usp=sharing)

The file `chairs_list.txt` maps the number of the file to its ShapeNet id, e.g. the chair numbered with 000000.png corresponds to id on line 0 of  `chairs_list.txt`.

This dataset also contains the preprocessed ProSketch dataset for simplicity. The ProSketch dataset, as well as the AmateurSketch dataset used in our paper for evaluation, can be found on the [SketchX](https://sketchx.eecs.qmul.ac.uk/downloads/) website.

## Running the code

The code has only been tested on Ubuntu 22.04. A GPU is required.


### Installation

After installing conda you can simply run:

```bash
conda env create -f environment.yml
conda activate SENS
```

Before running the code from the root of the repository, run

```export PYTHONPATH=.```
to avoid issues with module import.


### CLI

We provide an easy script for running the program on one input image in `run.py`. Use this to check your current installation, or for an offline skecth-to-shape pipeline.

`python run.py --input sketch.png`

An example sketch.png is provided. Results are saved in `assets/output`.

### Sketching interface

#### Launching

After following the installation instructions (be sure to run ```export PYTHONPATH=.```), from the root of the project run
```python ui_sketch/sketch_main.py```

You should see a window appear. There are two panels: the left one displays the resulting shape, the right one is where you can draw. Commands are to be found below.

#### Commands

You can draw on the right panel via **right-click**. To switch between erasing and drawing, click on the pencil icon. To generate a shape from the drawing, click on the shredder icon. You can rotate the mesh using left-click, and use the wheel to zoom-in/zoom-out. Click on the shape outline to get an outline rendering of the mesh. The outline rendering will correspond to the position and rotation of the mesh on the left, and the zoom level will influence the thickness of the stroke.

To select parts of the mesh, use the right-click. Selected parts are in orange. Use the hammer icon to regenerate the corresponding parts of the latent code. By modifying the drawing and clicking on the cube-with-orange-top icon, you will only modify the selected parts of the current mesh.

To save the current mesh to `assets/tmp/`, press ENTER. The program will also save your sketch at each stroke that you draw.


### Training

Download the dataset and extract it the `assets/data/` folder. You should have a folder `assets/data/dataset_chair_preprocess`. Then, simply launch
```python trainer_combined.py```

You can change the tag of the model via e.g.:

```python trainer_combined.py --tag chairs_personal_training```


## Citation

You can cite our work as
```
@article{
    Binninger:SENS:2024,
    author = {Alexandre Binninger and Amir Hertz and Olga Sorkine-Hornung and Daniel Cohen-Or and Raja Giryes},
    title = {{SENS}: Part-Aware Sketch-based Implicit Neural Shape Modeling},
    journal = {Computer Graphics Forum (Proceedings of EUROGRAPHICS 2024)},
    volume = {43},
    number = {2},
    year = {2024},
}
```


To cite [SPAGHETTI](https://amirhertz.github.io/spaghetti/), the shape decoder:
```
@article{
    hertz2022spaghetti, 
    title={SPAGHETTI: Editing Implicit Shapes Through Part Aware Generation},
    author={Hertz, Amir and Perel, Or and Giryes, Raja and Sorkine-Hornung, Olga and Cohen-Or, Daniel}, 
    journal={arXiv preprint arXiv:2201.13168}, 
    year={2022}
}
```