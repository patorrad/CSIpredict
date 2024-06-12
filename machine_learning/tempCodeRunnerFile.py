            ## Adding this line to track epoch
            if TENSORBOARD: writer.add_scalar("losses/val_loss", val_loss.item(), epoch)