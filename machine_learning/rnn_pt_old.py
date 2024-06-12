import datetime
import h5py
import numpy as np
import pandas as pd
from models import RNN, LSTM, GRU
import torch
import torch.nn as nn
from cprint import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from utils import calculate_metrics

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from dataset_consumer import DatasetConsumer

from utils import watts_to_dbm, get_scaler, dbm_to_watts

np.set_printoptions(threshold=sys.maxsize)


# for hidden_size in [8, 16, 32, 64, 128]:
for scaler_type in ['minmax', 'quantiletransformer-gaussian', 'quantiletransformer-uniform']: #, 'yeo-johnson']:
    DEBUG = True
    SCALER = scaler_type
    SAVE_PATH = './machine_learning/models/model.pth'
    NUM_PATHS = 100 #100000

    # Hyperparameters
    batch_size = 20000
    shuffle = True
    input_size = 128
    hidden_size = 64
    num_layers = 5
    output_size = 128
    sequence_length = 9
    learning_rate = 0.005
    dropout = .2
    num_epochs = 100
    batch_size = batch_size
    model_type = "gru"
    model_params = {'input_size': input_size,
                    'hidden_size' : hidden_size,
                    'num_layers' : num_layers,
                    'output_size' : output_size,
                    'dropout_prob' : dropout}

    DATASET = './machine_learning/data/dataset_0_5m_spacing.h5'
    d = DatasetConsumer(DATASET)
    d.print_info()

    # Scale mag data
    d.csi_mags = watts_to_dbm(d.csi_mags) # Convert to dBm
    scaler = get_scaler(SCALER) 
    cprint(f'SCALER: {scaler}')  # Print type of scaler used
    scaler.fit(d.csi_mags.T)
    d.csi_mags = d.scale(scaler.transform, d.csi_mags.T).T

    # Find paths
    d.csi_phases = d.unwrap(d.csi_phases)
    paths = d.generate_straight_paths(NUM_PATHS, 10)
    dataset_mag = d.paths_to_dataset_mag_only(paths)
    dataset_phase = d.paths_to_dataset_phase_only(paths)
    dataset_positions = d.paths_to_dataset_positions(paths)

    # # Convert 'split_sequences' to a PyTorch tensor
    dataset_mag = torch.from_numpy(dataset_mag)
    # Split dataset into train, val and test
    X_train, X_test, y_train, y_test = train_test_split(dataset_mag[:,:9,:], dataset_mag[:,9:10,:].squeeze(), train_size = 0.85, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 0.8, shuffle=False)

    # Dataset
    train = TensorDataset(X_train, y_train)
    validate = TensorDataset(X_val, y_val)
    test = TensorDataset(X_test, y_test)

    # Create a data loader
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    validate_dataloader = DataLoader(validate, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    # Loop through the data using the data loader
    for i, batch in enumerate(train_dataloader, 0):
        cprint(f'Batch {i} has size {batch[0].shape[0]}')  # Check the batch size


    def get_model(model, model_params):
        models = {
            "rnn": RNN,
            "lstm": LSTM,
            "gru": GRU,
        }
        return models.get(model.lower())(**model_params)

    layout = {
        "ABCDE": {
            "loss": ["Multiline", ["losses/running_train_loss", "losses/running_val_loss", "losses/test_loss"]],
            "accuracy": ["Multiline", ["accuracy_val/mae", "accuracy_val/rmse", "accuracy_val/r2"]],
        },
    }
    current = datetime.datetime.now()
    writer = SummaryWriter(f"runs/{model_type}_{num_epochs}_{num_layers}_{hidden_size}_{learning_rate}_{dropout}_{NUM_PATHS}_{batch_size}_{SCALER}_{current.month}-{current.day}-{current.hour}:{current.minute}")
    writer.add_custom_scalars(layout)

    # Create a simple RNN model
    model = get_model(model_type, model_params)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate adjustment
    # decay_lr_lambda = lambda epoch: 0.2 if optimizer.param_groups[0]['lr'] < 0.0001 else 0.915 ** (epoch // 5)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lr_lambda)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    # Training loop
    model = model.float()
    i = 0
    j = 0
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        running_val_loss = 0

        model.train()
        for batch in train_dataloader:
            sequences, targets = batch
            outputs = model(sequences.float())
            loss = criterion(outputs, targets.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            writer.add_scalar("losses/running_train_loss", loss.item(), i)
            i += 1

        model.eval()
        with torch.no_grad():
            for batch in validate_dataloader:
                sequences, targets = batch
                outputs = model(sequences.float())
                val_loss = criterion(outputs, targets.float())
                running_val_loss += val_loss.item()
                writer.add_scalar("losses/running_val_loss", val_loss.item(), j)

                # Create dataframe
                df_result = pd.DataFrame({
                    'value': targets.flatten(),  # flatten() is used to convert the arrays to 1D if they're not already
                    'prediction': outputs.flatten()
                })

                # Calcuate metrics
                result_metrics = calculate_metrics(df_result)
                writer.add_scalar("accuracy_val/mae", result_metrics['mae'], j)
                writer.add_scalar("accuracy_val/rmse", result_metrics['rmse'], j)
                writer.add_scalar("accuracy_val/r2", result_metrics['r2'], j)
                j += 1


        if epoch % 100 == 99:
            print(f'[{epoch + 1}, {num_epochs}] loss: {running_train_loss:.3f}')
            running_train_loss = 0.0

        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], i)
        scheduler.step(val_loss.item())

    # Save test dataset tensors
    torch.save(X_test, f'./machine_learning/models/X_test_{model_type}.pt')
    torch.save(y_test, f'./machine_learning/models/y_test_{model_type}.pt')

    ## Run Test dataset
    model.eval()

    test_loss = 0.0  # Initialize the test loss
    total_samples = 0  # Total samples in the test dataset

    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:  # Assuming you have a DataLoader for the test dataset
            sequences, targets = batch  # Get input sequences and their targets
            cprint.info(f'sequences {sequences.shape}')
            outputs = model(sequences.float())  # Make predictions
            predictions.append(outputs)
            loss = criterion(outputs, targets)  # Calculate the loss
            writer.add_scalar("losses/test_loss", loss.item(), total_samples)
            test_loss += loss.item()
            total_samples += 1
            print(f'total_samples {total_samples}')

    # Calculate the average test loss
    average_test_loss = test_loss / total_samples

    # Optionally, calculate and print other evaluation metrics if needed
    cprint.info(f"Average Test Loss: {average_test_loss:.4f}")

    # Create dataframe
    df_result = pd.DataFrame({
        'value': y_test.flatten(),  # flatten() is used to convert the arrays to 1D if they're not already
        'prediction': predictions[0].flatten()
    })

    # Calcuate metrics
    result_metrics = calculate_metrics(df_result)
    cprint.info(f'Results: {result_metrics}')


    if DEBUG:
        # Sanity check
        for i in range(10):
            # To use the trained model for prediction, you can pass new sequences to the model:
            rand = torch.randint(0, X_test.shape[0], (1,))
            new_input = X_test[rand,:]
            input_descaled_log = scaler.inverse_transform(new_input.squeeze().detach().numpy())
            input_descaled_linear = dbm_to_watts(input_descaled_log)
            ground_truth = y_test[rand,:]
            ground_truth_log = scaler.inverse_transform(ground_truth)
            ground_truth_linear = dbm_to_watts(ground_truth_log)

            # Prediction
            prediction = model(new_input.to(torch.float32))

            cprint.info(f'pred: {prediction.shape}')

            prediction_log = scaler.inverse_transform(prediction[0, :, :].detach().numpy())
            prediction_linear = dbm_to_watts(prediction_log)
            
            # Graphs
            fig, axs = plt.subplots(11, figsize=(10,10))
            plt.subplots_adjust(hspace=1.25)
            axs[0].plot(input_descaled_linear[4,:])
            axs[0].set_title("CSI Reading 5")
            axs[1].plot(input_descaled_linear[5,:])
            axs[1].set_title("CSI Reading 6")
            axs[2].plot(input_descaled_linear[6,:])
            axs[2].set_title("CSI Reading 7")
            axs[3].plot(input_descaled_linear[7,:])
            axs[3].set_title("CSI Reading 8")
            axs[4].plot(input_descaled_linear[8,:])
            axs[4].set_title("CSI Reading 9")
            axs[5].plot(ground_truth.squeeze())
            axs[5].set_title("Ground Truth Scaled")
            axs[6].plot(ground_truth_log.squeeze())
            axs[6].set_title("Ground Truth Log")
            axs[7].plot(ground_truth_linear.squeeze())
            axs[7].set_title("Ground Truth Linear")
            axs[8].plot(prediction.detach().numpy().squeeze())
            axs[8].set_title("Prediction")
            axs[9].plot(prediction_log.squeeze())
            axs[9].set_title("Prediction Log")
            axs[10].plot(prediction_linear.squeeze())
            axs[10].ticklabel_format(useOffset=False)
            axs[10].set_title("Prediction Linear")
            # plt.show()
            writer.add_figure(f'Comparison {i}', fig, global_step=0)
            plt.close(fig)
            

    # Close tensorboard writer
    writer.close()
    # Save model
    torch.save(model.state_dict(), SAVE_PATH)

    # Final learning rate
    final_learning_rate = optimizer.param_groups[0]['lr']
    cprint.ok(f'Final learning rate {final_learning_rate}')

# # NOTE Run "tensorboard --logdir runs" to see results