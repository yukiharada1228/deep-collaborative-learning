import os

import torch


def save_checkpoint(model, save_dir, epoch, is_best=False):
    state = {
        "epoch": epoch,
        "arch": model.__class__.__name__,
        "state_dict": model.state_dict(),
    }
    if is_best:
        path = os.path.join(save_dir, "best_checkpoint.pkl")
    else:
        path = os.path.join(save_dir, "checkpoint_epoch_%d.pkl" % epoch)
    torch.save(state, path)


def load_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
