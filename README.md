# HistoGAN
Experiments for different GAN problems related to histopathology image analysis

`requirements.txt` can be used to create an environment with the necessary packages via `pip install -r requirements` after creating and actviating a blank python virtualenv. OpenSlide needs to be installed separately and can be done so with the conda command `conda install -c conda-forge openslide`.

## Different experiment directories and contents
### Wasserstein_BiGAN
```
├── main.py
├── models.py
├── patch_dataset.py
├── preprocess.py
├── scratch_results
│   └── models
├── train.py
└── util.py
```
Code for (Wasserstein-)BiGAN image compression system to be trained on slide patch images, and ultimately used to create a three-dimensional compression of whole-slide images. `main.py` provides all available user-specified experimental settings (as well as default values for the CLI arguments), and an example call that will instantiate and train a BiGAN on the SVHN dataset is
```
python main.py --dataset svhn --image_size 32 --epochs 200
```
