import h5py
import torch
import torch.nn as nn
from cprint import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Specify the path to your HDF5 file
hdf5_file_path = '/Users/vibhasathishkumar/matlab_dataset_generator/dataset.h5' #matlab_dataset_generator\dataset.h5' #'dataset.h5'

# Open the HDF5 file for reading
with h5py.File(hdf5_file_path, 'r') as file:
    # Access the dataset you want to read
    csis_mag = file['csis_mag']  # Replace 'your_dataset_name' with the actual dataset name
    positions = file['positions']
    # Read the data into a NumPy array
    csis_mag = csis_mag[:]
    positions = positions[:]
    cprint.info(f'csis_mag {csis_mag.shape}')
    cprint.info(f'positions {positions.shape}')

# Convert the NumPy array to a PyTorch tensor
csis_mag = torch.from_numpy(csis_mag)
positions = torch.from_numpy(positions)
cprint.info(f'position {positions[:,790]}')

def bresenham_line(x1_pos, y1_pos, x2_pos, y2_pos, new_x_positions, new_y_positions, positions): # use the dataset somehow to cross check the x-values in the dataset? 
    dx = abs(x2_pos - x1_pos)
    dy = abs(y2_pos - y1_pos)
    x = 0
    y = 0
    x_curr = x1_pos
    x_prev = x1_pos
    p = 2 * dx - dy # P is the decision variable
    while (x <= x2_pos):
        # specific to our dataset where the x-values increment by ~10
        # y-values will increase according to the dataset, but the x-values increment without following the dataset,
        # is there a way to only note the y and x-values that are in a line and in the dataset?
        new_x_positions.append(x) 
        new_y_positions.append(positions[1,y].item())
        x += 1
        if (p < 0):
            p = p + 2 * dy
        else:
            p = p + 2 * dy - 2 * dx
            y += 1

new_x_positions = []
new_y_positions = []

bresenham_line(positions[0,0].item(), positions[1,0].item(), positions[0,50], positions[1,50], new_x_positions, new_y_positions, positions)

plt.plot(new_x_positions[:], new_y_positions[:],'.',color='b')
plt.show()

# Define the sequence length
sequence_length = 10

# Calculate the number of sequences that can be obtained
num_sequences = csis_mag.size(1) // sequence_length
print(num_sequences)
# Create a list to store the split sequences
split_sequences = []

# Loop through the input tensor and split it into sequences
# it splits it into sequences of 3 values
for i in range(num_sequences):
    start = i * sequence_length
    end = (i + 1) * sequence_length
    sequence = csis_mag[:, start:end]
    split_sequences.append(sequence)

# Convert 'split_sequences' to a PyTorch tensor if needed
split_sequences_tensor = torch.stack(split_sequences, dim=2)

# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data  # data is a list of 10-element sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index]
        return sequence

# Create a sample dataset (you should replace this with your own data)
# Assuming 'split_sequences_tensor' from the previous example
data = split_sequences_tensor.T

# Initialize the custom dataset
dataset = MyDataset(data)

# Define hyperparameters
batch_size = 15
shuffle = True

# Create a data loader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Loop through the data using the data loader
for batch in dataloader:
    # 'batch' will contain a batch of 10-element sequences
    print(batch.size())  # Check the batch size

# Define a simple RNN class
# Recurrent Neural Network
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through the RNN layer
        out, _ = self.rnn(x, h0)
        
        # Reshape the output for the fully connected layer
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Hyperparameters
input_size = 128
hidden_size = 64
num_layers = 1
output_size = 128
sequence_length = 10
learning_rate = 0.001
num_epochs = 1000

# Create a simple RNN model
model = SimpleRNN(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Sample input data
input_data = torch.randn(1, sequence_length, input_size)

# Sample target data
target_data = torch.randn(1, output_size)

# Training loop
for epoch in range(num_epochs):
    outputs = model(input_data)
    loss = criterion(outputs, target_data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# To use the trained model for prediction, you can pass new sequences to the model:
# new_input = torch.randn(1, sequence_length, input_size)
new_input = split_sequences_tensor.T[torch.randint(0, split_sequences_tensor.T.shape[0], (1,)),:,:]
# new_input = csis_mag[:,torch.randint(0, csis_mag.shape[1])]
# new_input = split_sequences_tensor.T[torch.randint(0, csis_mag.T.shape[0], (1,)),:,:]
# Prediction:
# prediction = model(new_input.to(torch.float32))
prediction = model(new_input.to(torch.float32))

# GRAPHS: 
new_input = new_input.squeeze()
cprint.warn(new_input[9,:].shape)
plt.plot(new_input[9,:]) #    WHAT IS THIS PLOTING? 
plt.show()

prediction = prediction.detach().numpy()
cprint.info(prediction)
cprint.info(prediction.size)
plt.plot(prediction.squeeze())
plt.show()

# cprint.info(new_input)
# cprint.warn(new_input.shape)
# plt.plot(new_input.squeeze())

# plt.plot(prediction.detach().numpy())

#plt.show()
# cprint.info(prediction)
