"""Pytorch Dataset

Class definition and an utility function.
"""
import numpy as np
from torch.utils.data import Dataset

from models import TRAIN_PERIODS

MEMMAP_SHAPES = {
    "train": [(40008, 178, 5), (40008, 178, 11), (40008, 39)],
    "val": [(815, 178, 5), (815, 178, 11), (815, 39)],
    "test": [(821, 178, 5), (821, 178, 11)]
}


class RecruitDataset(Dataset):
    def __init__(self, shapes, x, x_i, y=None, reference_dataset=None, train_periods=70, clip_low=-3, clip_high=3):
        """
        args
        ----
            x:
            y:   pass False for test dataset
            split_point: where to split data for encoder/decoder
        """
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.reference_dataset = reference_dataset
        self.series = np.memmap(
            x, mode="r", order="C", dtype="float64", shape=shapes[0]).__array__()
        self.series_i = np.memmap(
            x_i, mode="r", order="C", dtype="int16", shape=shapes[1]).__array__()
        self.train_periods = train_periods
        if y is None:
            self.is_train = False
        else:
            self.is_train = True
            self.y = np.memmap(
                y, mode="r", order="C", dtype="float64", shape=shapes[2]).__array__()
        self.fit_residual_series_stats()

    def __len__(self):
        return len(self.series)

    def fit_residual_series_stats(self, cols=[0, 1, 2, 3, 4]):
        data = self.series[:, :, cols]
        # means = np.mean(data[:, (self.train_periods-28):self.train_periods, :], axis=1)
        means = np.mean(data[:, :self.train_periods, :], axis=1)
        self.means = means
        if self.reference_dataset is not None:
            self.stds = self.reference_dataset.stds
        else:
            data = data - np.expand_dims(means, 1)
            self.stds = np.concatenate([
                np.std(data[:, :self.train_periods, :2].reshape(
                    -1, 2), axis=0),
                np.std(data[:, :, 2:].reshape(-1, 3), axis=0)
            ], axis=0)
            print(self.stds)
        self.mean_of_means = np.mean(means, axis=0)
        self.std_of_means = np.std(means, axis=0)

    def normalize_series(self, idx):
        residual_data = self.series[idx, :, :5]
        residual_data = (
            residual_data - self.means[idx, np.newaxis, :]) / self.stds[np.newaxis, :]
        # Set visitors to 0 when the store is closed
        residual_data[(self.series[idx, :, 0] == 0), 0] = 0
        return np.clip(np.nan_to_num(
            residual_data), self.clip_low, self.clip_high)

    def derive_features(self, idx):
        feat = np.zeros(5)
        cnt = 0
        # year 2 visitor mean
        feat[cnt] = (self.means[idx, 0] - self.mean_of_means[0]) / \
            self.std_of_means[0]
        cnt += 1
        # reserve from mean
        feat[cnt] = (self.means[idx, 1] - self.mean_of_means[1]) / \
            self.std_of_means[1]
        cnt += 1
        # reserve to mean
        feat[cnt] = (self.means[idx, 2] - self.mean_of_means[2]) / \
            self.std_of_means[2]
        cnt += 1
        # Precipitation mean
        feat[cnt] = (self.means[idx, 3] - self.mean_of_means[3]) / \
            self.std_of_means[3]
        cnt += 1
        # Temperature mean
        feat[cnt] = (self.means[idx, 4] - self.mean_of_means[4]) / \
            self.std_of_means[4]
        cnt += 1
        return np.nan_to_num(feat).astype("float32")

    def __getitem__(self, idx):
        numeric_series = self.normalize_series(idx)
        # print(idx, np.max(self.series_i[idx, :, :].__array__(), axis=0))
        if self.is_train:
            return (
                numeric_series,
                self.derive_features(idx),
                self.series_i[idx, :, :],
                # np.concatenate([
                #     self.series_i[idx, :, :],
                #     (self.series[idx, :, 0] == 0)[:, np.newaxis]
                # ], axis=1),
                self.means[idx],
                self.y[idx, :]
            )
        else:
            return (
                numeric_series,
                self.derive_features(idx),
                self.series_i[idx, :, :],
                # np.concatenate([
                #     self.series_i[idx, :, :],
                #     (self.series[idx, :, 0] == 0)[:, np.newaxis]
                # ], axis=1),
                self.means[idx]
            )


def read_dataset():
    x_train = "cache/xtrain_seq.npy"
    x_i_train = "cache/xtrain_i_seq.npy"
    y_train = "cache/ytrain_seq.npy"
    x_val = "cache/xval_seq.npy"
    x_i_val = "cache/xval_i_seq.npy"
    y_val = "cache/yval_seq.npy"
    x_test = "cache/xtest_seq.npy"
    x_i_test = "cache/xtest_i_seq.npy"

    train_dataset = RecruitDataset(
        MEMMAP_SHAPES["train"], x_train, x_i_train, y_train, train_periods=TRAIN_PERIODS)
    val_dataset = RecruitDataset(
        MEMMAP_SHAPES["val"], x_val, x_i_val, y_val, train_periods=TRAIN_PERIODS,
        reference_dataset=train_dataset)
    test_dataset = RecruitDataset(
        MEMMAP_SHAPES["test"], x_test, x_i_test, train_periods=TRAIN_PERIODS,
        reference_dataset=train_dataset)
    return train_dataset, val_dataset, test_dataset
