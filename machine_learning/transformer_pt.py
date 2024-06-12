## Based on Time Series Prediction with Transformers https://towardsdatascience.com/time-series-prediction-with-transformers-2b64478a4cbd
## Githu: https://github.com/hermanmichaels/transformer_example

import datetime

import torch
from torch.utils.data import DataLoader
import pandas as pd

from models import TransformerWithPE
from utils import (load_and_partition_RF_data, make_datasets, move_to_device,
                   split_sequence, visualize, EarlyStopping, calculate_metrics)

from torch.utils.tensorboard import SummaryWriter

from cprint import *

import os
import argparse

DATASET = 'dataset_0_5m_spacing.h5'

DEBUG = True
TENSORBOARD = True
SCALER = 'quantiletransformer-gaussian'
SAVE_PATH = './machine_learning/models/model.pth'
NUM_PATHS = 10000
PATH_LENGTH = 100
SEQ_LENGTH = 100
PRED_LENGTH = SEQ_LENGTH * 0.8

NUM_EX_FIGURES = 10

BS = 500
FEATURE_DIM = 128
NUM_HEADS = 8
NUM_EPOCHS = 100
NUM_VIS_EXAMPLES = 10
NUM_LAYERS = 2
LR = 0.01
D_MODEL = 512
E_LAYERS = NUM_LAYERS
D_LAYERS = NUM_LAYERS
D_FF = 2048
ATTN = 8
PATIENCE = 7

MODEL = 'transformer'
FEATURES = 1
CHECKPOINT = './checkpoints/'

setting = '{}_{}_ft{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}'.format(MODEL, DATASET, FEATURES, SEQ_LENGTH,
                                    PRED_LENGTH, D_MODEL, NUM_HEADS, E_LAYERS, D_LAYERS, D_FF, ATTN)
path = os.path.join(CHECKPOINT, setting)
if not os.path.exists(path):
    os.makedirs(path)


def main() -> None:
    # Load data and generate train and test datasets / dataloaders
    sequences, num_features, scaler = load_and_partition_RF_data("data.npz", SEQ_LENGTH, NUM_PATHS, PATH_LENGTH, DATASET, SCALER)
    cprint.ok(f'sequences.shape: {sequences.shape}')
    cprint.warn(f'num_features: {num_features}')
    train_set, test_set = make_datasets(sequences)
    train_loader, test_loader = DataLoader(
        train_set, batch_size=BS, shuffle=True
    ), DataLoader(test_set, batch_size=BS, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Initialize model, optimizer and loss criterion
    model = TransformerWithPE(
        num_features, num_features, FEATURE_DIM, NUM_HEADS, NUM_LAYERS
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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
            writer = SummaryWriter(f"runs/transformer_{NUM_EPOCHS}_{FEATURE_DIM}_{NUM_HEADS}_{PATH_LENGTH}_{NUM_PATHS}_{BS}_{SCALER}_{num_features}_{current.month}-{current.day}-{current.hour}:{current.minute}")
            writer.add_custom_scalars(layout)

    i_lr = 0
    j_lr = 0

    early_stopping = EarlyStopping(patience = PATIENCE, verbose=True)

    # Train loop
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            src, tgt, tgt_y = split_sequence(batch[0])
            src, tgt, tgt_y = move_to_device(device, src, tgt, tgt_y)
            # [bs, tgt_seq_len, num_features]
            pred = model(src, tgt)
            loss = criterion(pred, tgt_y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            if TENSORBOARD: writer.add_scalar("losses/running_train_loss", loss.item(), i_lr)
            i_lr += 1

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: "
            f"{(epoch_loss / len(train_loader)):.4f}"
        )

        # Evaluate model
        model.eval()
        eval_loss = 0.0
        infer_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                src, tgt, tgt_y = split_sequence(batch[0])
                src, tgt, tgt_y = move_to_device(device, src, tgt, tgt_y)

                # [bs, tgt_seq_len, num_features]
                pred = model(src, tgt)
                loss = criterion(pred, tgt_y)
                eval_loss += loss.item()
                if TENSORBOARD: 
                    writer.add_scalar("losses/running_val_loss", loss.item(), j_lr)
                    j_lr += 1
                # Run inference with model
                pred_infer = model.infer(src, tgt.shape[1])
                loss_infer = criterion(pred_infer, tgt_y)
                infer_loss += loss_infer.item()

                if idx < NUM_VIS_EXAMPLES:
                    if not TENSORBOARD:
                        writer = None
                    visualize(src, tgt, pred, pred_infer, j_lr, writer=writer)
                
                tgt_y_metrics = tgt_y.repeat(1,1,128)
                pred_infer_metrics = pred_infer.repeat(1,1,128)
                # Create dataframe
                df_result = pd.DataFrame({
                    'value': scaler.inverse_transform(tgt_y_metrics.reshape(-1,128)).flatten(),
                    'prediction': scaler.inverse_transform(pred_infer_metrics.reshape(-1,128)).flatten()
                })

                # Calcuate metrics
                result_metrics = calculate_metrics(df_result)
                if TENSORBOARD: 
                    writer.add_scalar("accuracy_val/mae", result_metrics['mae'], j_lr)
                    writer.add_scalar("accuracy_val/rmse", result_metrics['rmse'], j_lr)
                    writer.add_scalar("accuracy_val/r2", result_metrics['r2'], j_lr)

        avg_eval_loss = eval_loss / len(test_loader)
        avg_infer_loss = infer_loss / len(test_loader)

        print(f"Eval / Infer Loss on test set: {avg_eval_loss:.4f} / {avg_infer_loss:.4f}")
        if TENSORBOARD: writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], i_lr)
        scheduler.step(loss_infer.item())

        early_stopping(loss_infer, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()