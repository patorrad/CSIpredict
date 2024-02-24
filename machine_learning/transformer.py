## Based on influenza transformer from this repo https://github.com/KasperGroesLudvigsen/influenza_transformer

import utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn 
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
SCALER = 'quantiletransformer-gaussian'
SAVE_PATH = './machine_learning/models/model.pth'
NUM_PATHS = 500
PATH_LENGTH = 10

NUM_EX_FIGURES = 10

# Hyperparams
validation_set_size = 0.1
batch_size = 2
shuffle = True
num_epochs = 10

## Params
dim_val = 128 #512
n_heads = 2 #8
n_decoder_layers = 1 #4
n_encoder_layers = 1 #4
output_sequence_length = 1 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
enc_seq_len = PATH_LENGTH - output_sequence_length # length of input given to encoder
dec_seq_len = output_sequence_length # length of input given to decoder
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = True
learning_rate = 0.5
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
cprint.ok("Training data shape: {}".format(training_data.data[0:10,:])
          )
training_data = DataLoader(training_data, batch_size)
val_data = DataLoader(val_data, batch_size)

model = TimeSeriesTransformer(
    input_size=input_variables,
    dec_seq_len=enc_seq_len,
    batch_first=batch_first,
    # out_seq_len: int=58,
    # dim_val: int=512,  
    # n_encoder_layers: int=4,
    # n_decoder_layers: int=4,
    # n_heads: int=8,
    dropout_encoder = 0, 
    dropout_decoder = 0,
    dropout_pos_enc = 0,
    dim_feedforward_encoder = 128, #int=2048,
    dim_feedforward_decoder = 128, #int=2048,
    num_predicted_features=1
    )

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

criterion = torch.nn.MSELoss()

layout = {
    "ABCDE": {
        "loss": ["Multiline", ["losses/running_train_loss", "losses/running_val_loss", "losses/test_loss"]],
        "accuracy": ["Multiline", ["accuracy_val/mae", "accuracy_val/rmse", "accuracy_val/r2"]],
    },
}
current = datetime.datetime.now()
if TENSORBOARD:
        writer = SummaryWriter(f"runs/transformer_{num_epochs}_{dim_val}_{n_heads}_{enc_seq_len}_{NUM_PATHS}_{batch_size}_{SCALER}_{current.month}-{current.day}-{current.hour}:{current.minute}")
        writer.add_custom_scalars(layout)

running_train_loss = 0
i_lr = 0
j_lr = 0

# Early stop
best_val_loss = float('inf')
patience = 3  # Number of epochs to wait for improvement before stopping
epochs_no_improve = 0
early_stop = False

cprint.warn(next(iter(training_data)))
for epoch in range(num_epochs):
    model.train()
    # Iterate over all (x,y) pairs in training dataloader
    for i, (src, tgt, tgt_y) in enumerate(training_data):
        # zero the parameter gradients
        optimizer.zero_grad()

        # Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
        if batch_first == False:

            shape_before = src.shape
            src = src.permute(1, 0, 2)
            # print("src shape changed from {} to {}".format(shape_before, src.shape))

            shape_before = tgt.shape
            tgt = tgt.permute(1, 0, 2)
            # print("src shape changed from {} to {}".format(shape_before, tgt.shape))

            tgt_y = tgt_y.permute(1, 0)

        # Generate masks
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=output_sequence_length
            )

        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt))

        src_mask = utils.generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=enc_seq_len
            )
        # src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        print(src_mask)
        print(src_mask.shape)
        # src_mask = torch.zeros((1, 9))
        # print(src_mask)
        # print(src_mask.shape)
        # Make forecasts
        # cprint.ok(f'src shape: {src.shape}')
        # cprint.ok(f'src: {src}')
        # cprint.warn(f'tgt shape: {tgt.shape}')
        # cprint.warn(f'tgt: {tgt}')
        # cprint.err(f'tgt_y shape: {tgt_y.shape}')
        # cprint.err(f'tgt_y: {tgt_y}')
        # cprint.ok(f'src_mask shape: {src_mask.shape}')
        # cprint.ok(f'src_mask: {src_mask}')
        # cprint.warn(f'tgt_mask shape: {tgt_mask.shape}')
        # cprint.warn(f'tgt_mask: {tgt_mask}')
        prediction = model(src, tgt, src_mask, tgt_mask)
        cprint.info(f'prediction shape: {prediction.squeeze().shape}')
        cprint.info(f'prediction: {prediction.squeeze()}')
        cprint.info(f'tgt_y.shape: {tgt_y.shape}')
        cprint.info(f'tgt_y: {tgt_y}')
        # Compute and backprop loss
        loss = criterion(tgt_y.squeeze(), prediction.squeeze())
        running_train_loss += loss.item()
        
        loss.backward()

        # Take optimizer step
        optimizer.step()
        if TENSORBOARD: writer.add_scalar("losses/running_train_loss", loss.item(), i_lr)
        i_lr += 1

    print(f'[{epoch + 1}, {num_epochs}] loss: {running_train_loss:.3f}')
    running_train_loss = 0.0
    # Iterate over all (x,y) pairs in validation dataloader to do inference
    model.eval()

    with torch.no_grad():
        images = np.sort(np.random.choice(range(batch_size), size=NUM_EX_FIGURES)) #, replace=False))

        counter = 0
        for j, (src, tgt, tgt_y) in enumerate(val_data):
            # Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
            # if batch_first == False:
            #     shape_before = src.shape
            #     src = src.permute(1, 0, 2)
            # prediction = run_encoder_decoder_inference(
            #     model=model, 
            #     src=src, 
            #     forecast_window=output_sequence_length,
            #     batch_size=batch_size,
            #     device = 'cuda: 0',
            #     batch_first=batch_first
            #     )
            # cprint.ok(f'src shape: {src.shape}')
            # cprint.ok(f'src: {src}')
            # cprint.warn(f'tgt shape: {tgt.shape}')
            # cprint.warn(f'tgt: {tgt}')
            # cprint.err(f'tgt_y shape: {tgt_y.shape}')
            # cprint.err(f'tgt_y: {tgt_y}')
            # cprint.ok(f'src_mask shape: {src_mask.shape}')
            # cprint.ok(f'src_mask: {src_mask}')
            # cprint.warn(f'tgt_mask shape: {tgt_mask.shape}')
            # cprint.warn(f'tgt_mask: {tgt_mask}')
            # cprint.info(f'prediction shape: {prediction.shape}')
            # cprint.info(f'prediction: {prediction}')            
            # Plot the source data and prediction
            # if j in images:
            # time_pred = np.arange(0, PATH_LENGTH - output_sequence_length, 1)
            # Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
            if batch_first == False:

                shape_before = src.shape
                src = src.permute(1, 0, 2)
                # print("src shape changed from {} to {}".format(shape_before, src.shape))

                shape_before = tgt.shape
                tgt = tgt.permute(1, 0, 2)
                # print("src shape changed from {} to {}".format(shape_before, tgt.shape))

            # Generate masks
            tgt_mask = utils.generate_square_subsequent_mask(
                dim1=output_sequence_length,
                dim2=output_sequence_length
                )

            src_mask = utils.generate_square_subsequent_mask(
                dim1=output_sequence_length,
                dim2=enc_seq_len
                )
            # import pdb; pdb.set_trace()
            prediction = model(src, tgt, src_mask, tgt_mask)
            cprint.info(f'prediction shape: {prediction.shape}')
            cprint.info(f'prediction: {prediction}')  
            time_pred = np.arange(0, PATH_LENGTH, 1)
            time_future = np.arange(PATH_LENGTH - output_sequence_length, PATH_LENGTH, 1)
            fig = plt.figure()
            rand = np.random.randint(0,int(batch_size * validation_set_size))
            cprint.ok(f'rand: {rand}')
            cprint.ok(f'time_future: {time_future}')
            plt.plot(time_pred[:PATH_LENGTH - output_sequence_length], src[:, rand, 0].detach().numpy(), label='Source Data')
            plt.plot(time_pred[PATH_LENGTH - output_sequence_length:PATH_LENGTH], tgt_y[rand, :].detach().numpy(), label='Target', color='green', marker='.')
            plt.plot(time_future, prediction[:, rand, 0].detach().numpy(), label='Prediction', color='red', marker='.')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'Source Data - batch indx {rand}')
            plt.legend()
            # Show the plots
            if TENSORBOARD: writer.add_figure(f'Comparison {epoch}', fig, global_step=0)
            # plt.close()
            plt.show()
            
            val_loss = criterion(tgt_y, prediction)
            if TENSORBOARD: writer.add_scalar("losses/running_val_loss", val_loss.item(), j_lr)
            j_lr += 1

            # If the validation loss is lower than the best validation loss seen so far,
            # save the model and reset the epochs_no_improve counter
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
                epochs_no_improve = 0
            else:
                # If the validation loss didn't improve, increment the counter
                epochs_no_improve += 1

            # If the counter reaches the patience, stop the training
            if epochs_no_improve == patience:
                print('Early stopping!')
                early_stop = True
                break

    if TENSORBOARD: writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], i_lr)
    scheduler.step(val_loss.item())

    if early_stop:
        break

        # forecast_window = output_sequence_length
        # for i, (src, _, tgt_y) in enumerate(val_data):

            # prediction = run_encoder_decoder_inference(
            #     model=model, 
            #     src=src, 
            #     forecast_window=forecast_window,
            #     batch_size=src.shape[1],
            #     device = 'cuda: 0',
            #     batch_first=True
            #     )
            # cprint.warn(f'prediction: {prediction}')
            # cprint.ok(f'prediction shape: {prediction.shape}')
            # cprint.ok(f'src shape: {src.shape}')
            # cprint.ok(f'tgt_y shape: {tgt_y.shape}')
            # loss = criterion(tgt_y, prediction)




# i, batch = next(enumerate(training_data))

# src, trg, trg_y = batch
# cprint.ok("src shape: {}".format(src.shape))
# cprint.ok("trg shape: {}".format(trg.shape))    
# cprint.ok("trg_y shape: {}".format(trg_y.shape))

# # Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
# if batch_first == False:

#     shape_before = src.shape
#     src = src.permute(1, 0, 2)
#     print("src shape changed from {} to {}".format(shape_before, src.shape))

#     shape_before = trg.shape
#     trg = trg.permute(1, 0, 2)
#     print("src shape changed from {} to {}".format(shape_before, trg.shape))

# model = TimeSeriesTransformer(
#     input_size=input_variables,
#     dec_seq_len=enc_seq_len,
#     batch_first=batch_first,
#     num_predicted_features=1
#     )

# # Make src mask for decoder with size:
# # [batch_size*n_heads, output_sequence_length, enc_seq_len]
# src_mask = utils.generate_square_subsequent_mask(
#     dim1=output_sequence_length,
#     dim2=enc_seq_len
#     )

# # Make tgt mask for decoder with size:
# # [batch_size*n_heads, output_sequence_length, output_sequence_length]
# tgt_mask = utils.generate_square_subsequent_mask( 
#     dim1=output_sequence_length,
#     dim2=output_sequence_length
#     )

# output = model(
#     src=src,
#     tgt=trg,
#     src_mask=src_mask,
#     tgt_mask=tgt_mask
#     )

