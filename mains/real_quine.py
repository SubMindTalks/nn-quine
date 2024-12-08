import os
import sys
import json
import itertools
import pathlib
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch import nn

# Add the current directory to the Python path
sys.path.append(os.getcwd())

# Custom modules
from utils import saver, ml_logging, config as cfg, formulas
from models.real_nnquines import RealQuine


def load_config(config_file):
    """Load the configuration file."""
    with open(config_file) as f:
        configs = json.load(f)
    return configs


def setup_device(configs):
    """Setup the computational device (CPU/GPU)."""
    if configs.get("device", None) is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = configs["device"]
    return torch.device(device)


def build_model(configs, device):
    """Build and initialize the Real Quine model."""
    model = RealQuine(
        n_hidden=configs["n_hidden"],
        n_layers=configs["n_layers"],
        act_func=getattr(nn, configs["act_func"])
    ).to(device)
    return model


def setup_optimizer(model, configs):
    """Setup the optimizer and learning rate scheduler."""
    optimizer = getattr(torch.optim, configs.get("optimizer", "Adam"))(
        itertools.chain(*[model.parameters()]), lr=configs["learning_rate"]
    )

    lr_scheduler = configs.get("lr_scheduler", None)
    if lr_scheduler:
        cls_name = lr_scheduler.pop("class_name")
        lr_scheduler = getattr(torch.optim.lr_scheduler, cls_name)(
            optimizer, **lr_scheduler
        )
    return optimizer, lr_scheduler


def train_model(model, data, index_list, optimizer, configs, logger, device):
    """Train the Real Quine model."""
    print("Starting training...")
    for epoch in trange(configs["num_epochs"], desc="Epoch"):
        model.train()

        # Shuffle parameter indices
        random.shuffle(index_list)
        total_loss = 0.0
        avg_relative_error = 0.0

        optimizer.zero_grad()
        for pos, param_idx in enumerate(tqdm(index_list, leave=False)):
            idx_vector = data[param_idx]
            param = model.get_param(param_idx)
            pred_param = model(idx_vector)

            # Compute loss
            mse = (param - pred_param) ** 2
            total_loss += mse.item()
            avg_relative_error += formulas.relative_difference(pred_param.item(), param.item())

            # Backpropagation
            mse.backward()
            if (pos + 1) % configs["batch_size"] == 0 or pos + 1 == len(index_list):
                optimizer.step()
                optimizer.zero_grad()

        # Log training metrics
        logger.scalar_summary('mse_loss', total_loss / len(index_list), epoch)
        logger.scalar_summary('rel_error', avg_relative_error / len(index_list), epoch)

        # Save the model checkpoint periodically
        if epoch % configs.get("save_freq", 10) == 0:
            saver.save_checkpoint([model], logger.log_dir, epoch, optimizer=optimizer)

    # Save the final model
    saver.save_checkpoint([model], logger.log_dir, configs["num_epochs"], optimizer=optimizer)
    print("Training complete.")


def evaluate_model(model, data, index_list, device):
    """Evaluate the Real Quine model."""
    print("Evaluating...")
    model.eval()
    with torch.no_grad():
        for pos, param_idx in enumerate(index_list):
            idx_vector = data[param_idx]
            param = model.get_param(param_idx).item()
            pred_param = model(idx_vector).item()
            mse = (param - pred_param) ** 2
            print(
                f"Param #{pos}: True= {param:.3e} Pred= {pred_param:.3e} MSE= {mse:.3e} "
                f"REL_ERR= {formulas.relative_difference(param, pred_param):.3e}"
            )


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run Real Quine training")
    parser.add_argument("config_file", help="Path to the config file")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    args = parser.parse_args()

    # Load configs and setup paths
    configs = load_config(args.config_file)
    expt_name = cfg.get_expt_name(args.config_file, configs)
    log_dir = ml_logging.get_log_dir(expt_name)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = ml_logging.Logger(log_dir)

    # Initialize device, model, and optimizer
    device = setup_device(configs)
    model = build_model(configs, device)
    optimizer, lr_scheduler = setup_optimizer(model, configs)

    # Generate data for the quine
    data = torch.eye(model.num_params, device=device)
    index_list = list(range(model.num_params))

    # Train the model
    train_model(model, data, index_list, optimizer, configs, logger, device)

    # Optionally evaluate the model
    if args.eval:
        evaluate_model(model, data, index_list, device)
