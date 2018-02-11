"""Fit a transformer model"""
import os
import random
import logging
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformer.Optim import ScheduledOptim
from sacred import Experiment

from dataset import read_dataset
from bots import TransformerBot
from io_utils import export_validation, export_test


logging.basicConfig(level=logging.WARNING)

ex = Experiment('Transformer')
ex.add_source_file("preprocess.py")


@ex.named_config
def no_tf_2l():
    batch_size = 32
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.1,
        "d_model": 128,
        "d_inner_hid": 256,
        "n_layers": 2,
        "n_head": 4,
        "propagate": False
    }


@ex.config
def no_tf_2l_256():
    batch_size = 32
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.25,
        "d_model": 256,
        "d_inner_hid": 256,
        "n_layers": 2,
        "n_head": 4,
        "propagate": False
    }


@ex.named_config
def no_tf_3l_128():
    batch_size = 64
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.25,
        "d_model": 128,
        "d_inner_hid": 256,
        "n_layers": 3,
        "n_head": 4,
        "propagate": False
    }


@ex.named_config
def no_tf_3l_192():
    batch_size = 32
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.25,
        "d_model": 192,
        "d_inner_hid": 256,
        "n_layers": 3,
        "n_head": 4,
        "propagate": False
    }


@ex.named_config
def no_tf_1l():
    batch_size = 128
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.2,
        "d_model": 128,
        "d_inner_hid": 256,
        "n_layers": 1,
        "n_head": 8,
        "propagate": False
    }


@ex.named_config
def no_tf_3l():
    batch_size = 128
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.1,
        "d_model": 128,
        "d_inner_hid": 256,
        "n_layers": 3,
        "n_head": 2,
        "propagate": False
    }


@ex.named_config
def tf():
    batch_size = 128
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.05,
        "d_model": 128,
        "d_inner_hid": 256,
        "n_layers": 2,
        "n_head": 4,
        "train_warmup": 1000,
        "propagate": True
    }


@ex.automain
def main(batch_size, model_details, seed):
    train_dataset, val_dataset, test_dataset = read_dataset()
    print(len(train_dataset), len(test_dataset))
    print(f"SEED: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batches_per_epoch = len(train_dataset) // batch_size

    # Start Training
    bot = TransformerBot(
        train_dataset, test_dataset, val_dataset=val_dataset,
        n_layers=model_details.get("n_layers", 6),
        n_head=model_details.get("n_head", 8),
        d_model=model_details.get("d_model", 512),
        d_inner_hid=model_details.get("d_inner_hid", 1024),
        d_k=model_details.get("d_k", 32),
        d_v=model_details.get("d_v", 32),
        propagate=model_details.get("propagate", False),
        hdrop=model_details.get("hdrop", 0),
        edrop=model_details.get("edrop", 0),
        odrop=model_details.get("odrop", 0),
        avg_window=500,
        clip_grad=0.5,
        tf_warmup=int(batches_per_epoch),
        tf_decay=0.1 ** (1 / 6),
        tf_steps=batches_per_epoch // 200 * 100,
        tf_min=0.1,
        steps=38
    )

    param_groups = [
        {
            "params": bot.model.get_trainable_parameters(), "lr": .1
        }
    ]

    # optimizer = optim.Adam(param_groups)
    optimizer = optim.SGD(param_groups, momentum=0.95, nesterov=True)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, cooldown=10,
        threshold=2e-4,
        min_lr=[x["lr"] * 0.1 ** 3 for x in param_groups]
    )
    # scheduler = None

    # optimizer = ScheduledOptim(
    #     optim.Adam(
    #         bot.model.get_trainable_parameters(), betas=(0.9, 0.98), eps=1e-09),
    #     model_details.get("d_model", 512),
    #     model_details.get("train_warmup", 2000))
    # scheduler = None

    best_performers = bot.train(
        optimizer, batch_size=batch_size, n_epochs=30,
        log_interval=batches_per_epoch // 20,
        snapshot_interval=batches_per_epoch // 20 * 2,
        early_stopping_cnt=30,
        scheduler=scheduler,
        non_mono=-1
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    val_pred = bot.predict_avg(is_test=False, k=6).cpu().numpy()
    mask = val_dataset.y > 0
    score = np.sqrt(
        np.sum(np.square(val_dataset.y - val_pred) * (mask)) / np.sum(mask))
    print(val_pred.shape)
    export_validation("cache/preds/val/{}_{:.6f}_{}.csv".format(
        bot.name, score, timestamp), val_pred)

    test_pred = bot.predict_avg(is_test=True, k=6).cpu().numpy()
    print(test_pred.shape)
    export_test("cache/preds/test/{}_{:.6f}_{}.csv".format(
        bot.name, score, timestamp), test_pred)

    bot.logger.info("Score: {:.6f}".format(score))
    return score
