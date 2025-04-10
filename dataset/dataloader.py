import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F
from .data_argument import data_aug

def init_dataloader(cfg: DictConfig,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    dataset = utils.TensorDataset(
        final_timeseires[:train_length+val_length+test_length],
        final_pearson[:train_length+val_length+test_length],
        labels[:train_length+val_length+test_length]
    )

    train_dataset, val_dataset, test_dataset = utils.random_split(
        dataset, [train_length, val_length, test_length])
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               smri: torch.tensor,
                               )-> List[utils.DataLoader]:
    
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    split = StratifiedShuffleSplit(n_splits=1, test_size=val_length + test_length)
    for train_index, test_valid_index in split.split(final_timeseires, labels):
        final_timeseires_train = final_timeseires[train_index]
        final_pearson_train = final_pearson[train_index]
        labels_train = labels[train_index]
        smri_train = smri[train_index]

        final_timeseires_val_test = final_timeseires[test_valid_index]
        final_pearson_val_test = final_pearson[test_valid_index]
        labels_val_test = labels[test_valid_index]
        smri_val_test = smri[test_valid_index]

    split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length)
    for test_index, valid_index in split2.split(final_timeseires_val_test, labels_val_test):
        final_timeseires_test = final_timeseires_val_test[test_index]
        final_pearson_test = final_pearson_val_test[test_index]
        labels_test = labels_val_test[test_index]
        smri_test = smri_val_test[test_index]

        final_timeseires_val = final_timeseires_val_test[valid_index]
        final_pearson_val = final_pearson_val_test[valid_index]
        labels_val = labels_val_test[valid_index]
        smri_val = smri_val_test[valid_index]

    if cfg.dataset.expand:
        final_timeseires_train, final_pearson_train, labels_train, smri_train = data_aug(final_timeseires_train, final_pearson_train, labels_train, smri_train)
    
    
    train_dataset = utils.TensorDataset(
        final_timeseires_train,
        final_pearson_train,
        labels_train,
        smri_train,
    )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val, smri_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test, smri_test
    )

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.val_batch_size, shuffle=False, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.test_batch_size, shuffle=False, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]
