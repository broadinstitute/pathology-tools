import tensorflow as tf
import os 
import argparse
from data_manipulation.data import Data
from models.generative.gans.PathologyGAN import PathologyGAN


parser = argparse.ArgumentParser(description='PathologyGAN trainer.')
parser.add_argument('--epochs', dest='epochs', type=int, default=45, help='Number epochs to run: default is 45 epochs.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='Batch size, default size is 64.')
parser.add_argument('--model', dest='model', type=str, default='PathologyGAN', help='Model name.')
parser.add_argument('--checkpoint', dest='checkpoint', required=False, help='Path to pre-trained weights (.ckt) of PathologyGAN.')
parser.add_argument('--main_path', dest='main_path', required=True, help='Main path for output data')
parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset name or path name for he slide h5 dataset')
parser.add_argument('--input_img_dim', dest='input_img_dim', type=int, default=224, help='Dimension of input images (used for network instantiation)')
# parser.add_argument('--monitor_FID', action='store_true', help='Bool flag to trigger FID monitering during training')
# parser.add_argument('--generator_dataset', dest='use_generator', type=bool,
#                     default=False, help='Flag for using alternate dataset object')
args = parser.parse_args()
# debug
print(f'CLI args = {args}')
epochs = args.epochs
batch_size = args.batch_size
model = args.model
checkpoint = args.checkpoint
input_img_dim = args.input_img_dim
# extra metrics to monitor during training
# TODO: need to debug the image processing from training method to FID calculation step
track_FID = False #args.monitor_FID

# requiring main_path be taken as input; feels dangerous to formulate it automatically (i.e., higher risk of collision?)
main_path = args.main_path #os.path.dirname(os.path.realpath(__file__))
dbs_path = os.path.dirname(os.path.realpath(__file__))

# Dataset information.
data_out_path = os.path.join(main_path, 'data_model_output')
data_out_path = os.path.join(data_out_path, model)
image_width = input_img_dim  #448
image_height = input_img_dim  #448
image_channels = 3
dataset = args.dataset #'tcga', 'vgh_nki', or '{dataset_dir}/{filename}.h5'
# TODO: remove use of use_generator (or implement it), currently not being used
use_generator = False# args.generator_dataset
marker = 'he'
name_run = 'h%s_w%s_n%s' % (image_height, image_width, image_channels)
data_out_path = '%s/%s' % (data_out_path, name_run)

# Hyperparameters.
learning_rate_g = 1e-4
learning_rate_d = 1e-4
beta_1 = 0.5
beta_2 = 0.9
# Setting restore to True so the training method looks for a model checkpoint at data_out_path/checkpoints to start training from
restore = checkpoint is not None #False
print(f'run_pathgan.py called with restore={restore}')

# Model
layers = 5
z_dim = 200
alpha = 0.2
n_critic = 5
gp_coeff = .65
use_bn = False
loss_type = 'relativistic gradient penalty'

# setting labels flag to False if we give 'tcga' as our dataset (because those don't have labels)
data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels,
            batch_size=batch_size, project_path=dbs_path, labels=(dataset == 'vgh_nki'), use_generator_dataset=use_generator,
            # optionally set to True if there are matching {path}_validation.h5, {path}_test.h5 datasets
            # to go along with the {path}_train.h5 training dataset
            validation=False, test=False)

with tf.Graph().as_default():
    pathgan = PathologyGAN(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1,
                           learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2,
                           n_critic=n_critic, gp_coeff=gp_coeff, loss_type=loss_type, model_name=model)
    losses = pathgan.train(epochs, data_out_path, data, restore, print_epochs=10, n_images=10, show_epochs=None,
                           check=checkpoint, track_FID=track_FID)
