import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from models import RNN, LSTM, GRU
from matplotlib import pyplot as plt
from cprint import *
import pandas as pd
from utils import calculate_metrics

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

# Load test data
X_test = torch.load(f'./machine_learning/models/X_test_{model_type}.pt')
y_test = torch.load(f'./machine_learning/models/y_test_{model_type}.pt')

# Hyperparameters
input_size = 128
hidden_size = 64
num_layers = 5
output_size = 128
sequence_length = 62
learning_rate = 0.005
dropout = .2
num_epochs = 100
batch_size = X_test.shape[0]
model_params = {'input_size': input_size,
                'hidden_size' : hidden_size,
                'num_layers' : num_layers,
                'output_size' : output_size,
                'dropout_prob' : dropout}

# Create a simple RNN model
model = get_model(model_type, model_params)
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()

test = TensorDataset(X_test, y_test)
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
cprint.info(f'Ground Truth: {y_test.shape}')
# Create dataframe
df_result = pd.DataFrame({
    'value': y_test.flatten(),  # flatten() is used to convert the arrays to 1D if they're not already
    'prediction': predictions[0].flatten()
})

# Calcuate metrics
result_metrics = calculate_metrics(df_result)
cprint.info(f'Results: {result_metrics}')

for i in range(5):
        # To use the trained model for prediction, you can pass new sequences to the model:
        # new_input = split_sequences_tensor[torch.randint(0, split_sequences_tensor.shape[0], (1,)),:,:]
        rand = torch.randint(0, X_test.shape[0], (1,))
        new_input = X_test[rand,:]
        ground_truth = y_test[rand,:]
        # Prediction
        prediction = model(new_input.to(torch.float32))

        # Graphs
        fig, axs = plt.subplots(5)
        new_input = new_input.squeeze()
        # cprint.warn(f'validation shape {new_input[8,:].shape}')
        axs[0].plot(new_input[6,:])
        axs[0].set_title("CSI Reading 7")
        axs[1].plot(new_input[7,:])
        axs[1].set_title("CSI Reading 8")
        axs[2].plot(new_input[8,:])
        axs[2].set_title("CSI Reading 9")

        prediction = prediction.detach().numpy()
        # writer.add_figure(f'Comparison {i}', fig, global_step=0)
        # plt.close(fig)
        # plt.show()
        
        # prediction = scaler.inverse_transform(prediction)
        axs[3].plot(ground_truth.squeeze())
        axs[3].set_title("Ground Truth")
        axs[4].plot(prediction.squeeze())
        axs[4].set_title("Prediction")

        # writer.add_figure(f'Comparison {i}', fig, global_step=0)
        # plt.close(fig)
        plt.show()