import os
from data_manipulation.dataset import Dataset, Generator_Dataset


class Data:
    def __init__(self, dataset, marker, patch_h, patch_w, n_channels, batch_size, project_path=os.getcwd(),
                 thresholds=(), labels=False, empty=False, use_generator_dataset=False, validation=False, test=False):
        # Adding the `use_generator_dataset` flag to trigger the alternate dataset class that uses a tensorflow
        # generator object to more efficiently open/traverse the hdf5 dataset file

        # Directories and file name handling.
        # # if an hdf5 file is given as the dataset input, use that path directly, o/w construct from experiment descriptors
        # debug
        print(f'INPUT DATASET TO DATA(): {dataset}')
        if dataset[-3:] == '.h5':
            self.hdf5_train = dataset
        else:
            self.dataset = dataset
            self.marker = marker
            self.dataset_name = '%s_%s' % (self.dataset, self.marker)
            print(f'self.dataset_name = {self.dataset_name}')
            relative_dataset_path = os.path.join(self.dataset, self.marker)
            print(f'relative_dataset_path = {relative_dataset_path}')
            relative_dataset_path = os.path.join('dataset', relative_dataset_path)
            print(f'relative_dataset_path = {relative_dataset_path}')
            relative_dataset_path = os.path.join(project_path, relative_dataset_path)
            print(f'relative_dataset_path = {relative_dataset_path}')
            self.pathes_path = os.path.join(relative_dataset_path, 'patches_h%s_w%s' % (patch_h, patch_w))
            print(f'self.pathes_path = {self.pathes_path}')
            self.hdf5_train = os.path.join(self.pathes_path, 'hdf5_%s_train.h5' % self.dataset_name)

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels
        self.batch_size = batch_size

        self.training = None
        if os.path.isfile(self.hdf5_train):
            print(f'SETTING DATA.TRAINING ATTRIBUTE TO {self.hdf5_train}')
            if use_generator_dataset:
                self.training = Generator_Dataset(dataset)
            else:
                self.training = Dataset(self.hdf5_train, patch_h, patch_w, n_channels, batch_size=batch_size, thresholds=thresholds, labels=labels, empty=empty)
        else:
            print(f'os.path.isfile(self.hdf5_train) == False')

        # Validation dataset, some datasets work with those.
        if validation:
            self.hdf5_validation = os.path.join(self.pathes_path, 'hdf5_%s_validation.h5' % self.dataset_name)
            self.validation = None
            if os.path.isfile(self.hdf5_validation):
                self.validation = Dataset(self.hdf5_validation, patch_h, patch_w, n_channels, batch_size=batch_size, thresholds=thresholds, labels=labels, empty=empty)

        # Test dataset
        if test:
            self.hdf5_test = os.path.join(self.pathes_path, 'hdf5_%s_test.h5' % self.dataset_name)
            self.test = None
            if os.path.isfile(self.hdf5_test):
                self.test = Dataset(self.hdf5_test, patch_h, patch_w, n_channels, batch_size=batch_size, thresholds=thresholds, labels=labels, empty=empty)
