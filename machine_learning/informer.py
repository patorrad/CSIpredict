## Based on influenza transformer from this repo https://github.com/KasperGroesLudvigsen/influenza_transformer

import utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import datetime
from models import TimeSeriesTransformer
import numpy as np
import dataset_transformer as ds

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from utils import calculate_metrics, TransformerDataset, run_encoder_decoder_inference

from dataset_consumer import DatasetConsumer

from utils import watts_to_dbm, get_scaler, dbm_to_watts

from cprint import *

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

DEBUG = True
TENSORBOARD = False
SCALER = 'minmax'
SAVE_PATH = './machine_learning/models/model.pth'
NUM_PATHS = 500
PATH_LENGTH = 100

NUM_EX_FIGURES = 10

# Hyperparams
validation_set_size = 0.1
batch_size = 10
shuffle = True
num_epochs = 10

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
output_sequence_length = 3 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
enc_seq_len = PATH_LENGTH - output_sequence_length # length of input given to encoder
dec_seq_len = output_sequence_length # length of input given to decoder
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False
learning_rate = 0.005
dropout = .2

# Define input variables 
# exogenous_vars = [] # should contain strings. Each string must correspond to a column name
# input_variables = [target_col_name] + exogenous_vars
# target_idx = 0 # index position of target in batched trg_y

# input_size = len(input_variables)
input_variables = 1 # Univariate

DATASET = '../dataset_0_5m_spacing.h5'
# DATASET = 'dataset_0_5m_spacing.h5'
d = DatasetConsumer(DATASET)
d.print_info()

# Scale mag data
d.csi_mags = watts_to_dbm(d.csi_mags) # Convert to dBm
scaler = get_scaler('minmax')
scaler.fit(d.csi_mags.T)
d.csi_mags = d.scale(scaler.transform, d.csi_mags.T).T

# Find paths
d.csi_phases = d.unwrap(d.csi_phases)
paths = d.generate_straight_paths(NUM_PATHS, PATH_LENGTH)
dataset_mag = d.paths_to_dataset_mag_only(paths)
dataset_phase = d.paths_to_dataset_phase_only(paths)
dataset_positions = d.paths_to_dataset_positions(paths)

# Generate a permutation of indices
indices = np.random.permutation(dataset_mag.shape[0])
# Use the indices to shuffle the first dimension of dataset_mag
dataset_mag = dataset_mag[indices]
# Split the dataset into training and val sets
split_idx = int(dataset_mag.shape[0] * (1 - validation_set_size))
train_mag, val_mag = dataset_mag[:split_idx], dataset_mag[split_idx:]
# # Convert 'split_sequences' to a PyTorch tensor
train_mag = torch.from_numpy(train_mag)
val_mag = torch.from_numpy(val_mag)
train_mag = np.expand_dims(train_mag[:,:,0].flatten(), axis=1)
val_mag = np.expand_dims(val_mag[:,:,0].flatten(), axis=1)
train_indices = np.arange(0, train_mag.shape[0], PATH_LENGTH)
val_indices = np.arange(0, val_mag.shape[0], PATH_LENGTH)
train_window_view = np.lib.stride_tricks.sliding_window_view(train_indices, (2,))
val_window_view = np.lib.stride_tricks.sliding_window_view(val_indices, (2,))
training_indices = [tuple(window) for window in train_window_view]
val_indices = [tuple(window) for window in val_window_view]
cprint.err(f'window_view shape: {train_window_view.shape}')
training_data = ds.TransformerDataset(
    data=torch.tensor(train_mag).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )
val_data = ds.TransformerDataset(
    data=torch.tensor(val_mag).float(),
    indices=val_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )
cprint.ok("Training data shape: {}".format(training_data.data.shape))
training_data = DataLoader(training_data, batch_size)
val_data = DataLoader(val_data, batch_size)