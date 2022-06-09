# HistoGAN
Experiments for different GAN problems related to histopathology image analysis

## Pathology-GAN
```
├── CRImage
├── data_manipulation
├── evaluation_automated_sweep
├── find_nearest_neighbors.py
├── generate_fake_samples.py
├── generate_image_interpolation.py
├── high_d_exemplar.pkl
├── low_d_exemplar.pkl
├── models
├── quantify_model_pipeline.py
├── real_features.py
#------ To download and add ------
├── dataset
    └── vgh_nki (download from https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw) 
├── data_model_output
    └── PathologyGAN
        └── h224_w224_n3_zdim_200
            └── checkpoints
                ├── PathologyGAN.ckt.data-00000-of-00001 (download tar file from https://figshare.com/s/0a31)
                ├── PathologyGAN.ckt.index
                └── PathologyGAN.ckt.meta
#---------------------------------
└── run_pathgan.py
```
### virtualenv
`requirements.txt` can be used to create an environment via `pip install -r requirements.txt` after creating and activating a blank python (version 3.6.8) virtualenv.

### Usage
This code is roughly equivalent to this repository (https://github.com/AdalbertoCq/Pathology-GAN) with slight changes to allow
for simple generation of example images that interpolate from specified latent vectors.

`generate_image_interpolation.py` can be used to generate a set of images that interpolates between two input latents with the following call:
```
python ./generate_image_interpolation.py --num_samples 100 --z_dim 200 --checkpoint data_model_output/PathologyGAN/h224_w224_n3_zdim_200/checkpoints/PathologyGAN.ckt --exemplar1 low_d_exemplar.pkl --exemplar2 high_d_exemplar.pkl
```
This calls a modified version of the `generate_fake_samples.py` script with added logic to interpolate between specified exemplars. The script
will expect pickled numpy arrays containing vectors of the same dimension as `z_dim`. The exemplars given are the 200-dimensional
representations corresponding to the following images:
high_d_exemplar.pkl             |  low_d_exemplar.pkl
:-------------------------:|:-------------------------:
![](Pathology-GAN/evaluation_automated_sweep/gen_0_alpha_0.png) | ![](Pathology-GAN/evaluation_automated_sweep/gen_99_alpha_100.png)

... and the python command given above will generate an `evaluation` directory whose contents should be the same as those given in `evaluation_automated_sweep`.
![](Pathology-GAN/evaluation_automated_sweep/img_sweep.png)

## Wasserstein_BiGAN
```
├── main.py
├── models.py
├── patch_dataset.py
├── preprocess.py
├── train.py
└── util.py
```
### Conda environment
`environment.yml` can be used to create a conda virtual environment with the necessary packages with `conda env create --file environment.yml`
### virtualenv
`requirements.txt` can be used, and OpenSlide needs to be installed separately and can be done so with the conda command `conda install -c conda-forge openslide`.

Code for (Wasserstein-)BiGAN image compression system to be trained on slide patch images, and ultimately used to create a three-dimensional compression of whole-slide images. `main.py` provides all available user-specified experimental settings (as well as default values for the CLI arguments), and an example call that will instantiate and train a BiGAN on the SVHN dataset is
```
python main.py --dataset svhn --image_size 32 --epochs 200
```
*NB: you'll need to create results directories before executing training*
