# TODO: DELETE THIS SCRIPT, GOING TO USE DIFFERENT FID PIPELINE

import tensorflow as tf
import numpy as np
from scipy import linalg
from models.score.utils import *
import argparse


def frechet_inception_distance(x_features, y_features, batch_size, sqrt=False):
	batch_scores = list()
	batches = int(x_features.shape.as_list()[0]/batch_size)
	for i in range(batches):
		if batches-1 == i:
			x_features_batch = x_features[i*batch_size: , :]
			y_features_batch = y_features[i*batch_size: , :]
		else:
			x_features_batch = x_features[i*batch_size : (i+1)*batch_size, :]
			y_features_batch = y_features[i*batch_size : (i+1)*batch_size, :]

		samples = x_features_batch.shape.as_list()[0]
		x_feat = tf.reshape(x_features_batch, (samples, -1))
		y_feat = tf.reshape(y_features_batch, (samples, -1))

		x_mean = tf.reduce_mean(x_feat, axis=0)
		y_mean = tf.reduce_mean(y_feat, axis=0)

		# Review this two lines.
		x_cov = covariance(x_feat)
		y_cov = covariance(y_feat)

		means = dot_product(x_mean, x_mean) + dot_product(y_mean, y_mean) - 2*dot_product(x_mean, y_mean)
		cov_s = linalg.sqrtm(tf.matmul(x_cov, y_cov), True)
		cov_s = cov_s.real
		covas = tf.trace(x_cov + y_cov - 2*cov_s)

		fid = means + covas
		if sqrt:
			fid = tf.sqrt(fid)
		batch_scores.append(np.array(fid))
	return np.mean(batch_scores), np.std(batch_scores)

# TODO: Implement InceptionV3 feature extraction and real/synthetic sample set generation
## --> (score.py -> tfgan.eval.frechet_classifier_distance_from_activations() seems to not be available)


def get_inception_embeddings(real_samples, synth_samples):
	pass


def generate_datasets(real_data, model=None, synthetic_data=None, num_samples=10000):
	# Given: real dataset, and either synth. dataset or trained model, and target number of sampels
	# --> if synth. data is given, we draw min(|real_data|, |synth_data|, num_samples) from that and real_data
	# ---> if model is given, we generate min(|real_data|, num_samples)
	pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='FID calculation script')
	parser.add_argument('--model', required=False, help='Path to trained model checkpoint used to generate synthetic samples')
	parser.add_argument('--synthetic_data', required=False, help='Path to synthetic dataset that can be provided if the '
																 'trained model was already used to generate samples')
	# ... more CLI arguments ...
	args = parser.parse_args()

	if args.model in None and args.synthetic_data is None:
		raise ValueError('Must provide either a trained model or a synthetic dataset')


