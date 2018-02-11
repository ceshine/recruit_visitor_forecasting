from datetime import date, timedelta
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

VAL_IDX_FILEPATH = "cache/val_seq_id.pkl"
TEST_IDX_FILEPATH = "cache/test_id.pkl"

VAL_START_DATE = date(2017, 3, 12)
TEST_START_DATE = date(2017, 4, 23)


def transform_predictions(pred, idx_filepath, start_date):
    idx_store_item = pd.read_pickle(idx_filepath)
    df_preds = pd.DataFrame(
        pred, index=idx_store_item,
        columns=pd.date_range(start_date, periods=39)
    ).stack().to_frame("visitors")
    df_preds.index.set_names(["air_store_id", "date"], inplace=True)
    df_preds["visitors"] = np.clip(np.expm1(df_preds.visitors), 0, 1e4)
    return df_preds.reset_index()


def create_base_dir(filepath):
    base_dir = PurePosixPath(filepath).parent
    Path(base_dir).mkdir(parents=True, exist_ok=True)


def export_validation(filepath, val_pred):
    create_base_dir(filepath)
    df_preds = transform_predictions(
        val_pred, VAL_IDX_FILEPATH, VAL_START_DATE)
    df_preds.to_csv(filepath, float_format="%.6f", index=False)


def export_test(filepath, test_pred):
    create_base_dir(filepath)
    df_preds = transform_predictions(
        test_pred, TEST_IDX_FILEPATH, TEST_START_DATE)
    df_preds["id"] = df_preds["air_store_id"].str.cat(
        df_preds["date"].astype(str), "_")
    sample_submission = pd.read_csv("data/sample_submission.csv")
    submission = df_preds[["id", "visitors"]].merge(
        sample_submission[["id"]], how="right").fillna(0)
    submission[["id", "visitors"]].to_csv(
        filepath, float_format="%.6f", index=False)
