import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from models import RNN, LSTM, GRU
from matplotlib import pyplot as plt
from cprint import *
import pandas as pd
from utils import calculate_metrics
import numpy as np

SAVE_PATH = './models'
SAVE_PATH = './machine_learning/models/model.pth'

def get_model(model, model_params):
    models = {
        "rnn": RNN,
        "lstm": LSTM,
        "gru": GRU,
    }
    return models.get(model.lower())(**model_params)

model_type = "gru"

# Load collected data
#df = pd.read_csv('./machine_learning/collected_data/mags_ts_LLTF.csv')
df = pd.read_csv('./machine_learning/collected_data/mags_ts_LLTF.csv')
data = torch.tensor(df.values) / 100

# s = torch.cat((data[:49,:], data[:49,:]), 0)
# t = torch.cat((data, data), 1)
# t = torch.cat((t, t[:,:18]), 1)
# t = t.unsqueeze(0)

# test = TensorDataset(t[:,:49,:], t[:,49:,:])

test = data.unsqueeze(0)
test = TensorDataset(test[:,:49,:], test[:,49:,:])

X_test = torch.load(f'./machine_learning/models/X_test_{model_type}.pt')
y_test = torch.load(f'./machine_learning/models/y_test_{model_type}.pt')

# Hyperparameters
FREQ_BINS = 55
input_size = FREQ_BINS
hidden_size = 64
num_layers = 5
output_size = FREQ_BINS
sequence_length = 54
learning_rate = 0.005
dropout = .2
num_epochs = 100
batch_size = 1
NUM_PREDICTIONS = 5
PATH_LENGTH = sequence_length
model_params = {'input_size': input_size,
                'hidden_size' : hidden_size,
                'num_layers' : num_layers,
                'output_size' : output_size,
                'dropout_prob' : dropout,
                'num_pred' : NUM_PREDICTIONS}

# Create a simple RNN model
model = get_model(model_type, model_params)
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()


#test = TensorDataset(X_test, y_test)
shuffle = False
# Assume `test_data` is your test data
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        # Assume `inputs` is your input data
        # inputs = batch
        # outputs = model(inputs)
        sequences, targets = batch  # Get input sequences and their targets
        outputs = model(sequences.float())  # Make predictions
        predictions.append(outputs)
        cprint.info(f'Predictions: {predictions[0].shape}')
        # Graphs
        fig, axs = plt.subplots(1)
        time_pred = np.arange(PATH_LENGTH - NUM_PREDICTIONS, PATH_LENGTH, 1)

        axs.plot(np.concatenate((sequences.squeeze()[:,], targets.squeeze()[:,])))
        axs.plot(time_pred, predictions[0].detach().numpy().squeeze()[:NUM_PREDICTIONS,], marker='.',  color='orange')
        axs.set_xlabel("Time Step")
        axs.set_ylabel("Magnitude of CSI")
        axs.set_title("Ground Truth with Prediction")

        plt.show()
    # for batch in test_dataloader:
    #     # Assume `inputs` is your input data
    #     # inputs = batch
    #     # outputs = model(inputs)
    #     sequences, targets = batch  # Get input sequences and their targets
    #     outputs = model(sequences.float())  # Make predictions
    #     predictions.append(outputs)
    #     cprint.info(f'Predictions: {predictions[0].shape}')
    #     # Graphs
    #     fig, axs = plt.subplots(1)
    #     time_pred = np.arange(PATH_LENGTH - NUM_PREDICTIONS, PATH_LENGTH, 1)

    #     axs.plot(np.concatenate((sequences.squeeze()[:,0], targets.squeeze()[:,0])))
    #     axs.plot(time_pred, predictions[0].detach().numpy().squeeze()[:NUM_PREDICTIONS,0], marker='.',  color='orange')
    #     axs.set_title("Ground Truth with Prediction")

    #     plt.show()


# Create dataframe
#df_result = pd.DataFrame({
#    'value': y_test.flatten(),  # flatten() is used to convert the arrays to 1D if they're not already
#    'prediction': predictions[0].flatten()
#})

# Calcuate metrics
#result_metrics = calculate_metrics(df_result)
#cprint.info(f'Results: {result_metrics}')

for i in range(1):
        # To use the trained model for prediction, you can pass new sequences to the model:
        # new_input = split_sequences_tensor[torch.randint(0, split_sequences_tensor.shape[0], (1,)),:,:]
        rand = torch.randint(0, X_test.shape[0], (1,))
        new_input = X_test[rand,:]
        ground_truth = y_test[rand,:]
        # Prediction
        prediction = model(new_input.to(torch.float32))

        # Graphs
        fig, axs = plt.subplots(1)
        new_input = new_input.squeeze()
        # cprint.warn(f'validation shape {new_input[8,:].shape}')
        axs[0].plot(new_input[6,:])
        axs[0].set_title("CSI Reading 7")

        plt.show()