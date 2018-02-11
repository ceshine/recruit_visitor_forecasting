import pathlib
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn import preprocessing
from joblib import Memory

memory = Memory(cachedir="cache/", verbose=1)

LOOKBACK = 140


@memory.cache
def read_data():
    pathlib.Path("cache/").mkdir(parents=True, exist_ok=True)

    data = {
        'tra': pd.read_csv(
            'data/air_visit_data.csv', parse_dates=["visit_date"],
            converters={'visitors': lambda u: np.log1p(
                float(u)) if u != "" and float(u) > 0 else 0},
            engine="c"
        ).set_index(["air_store_id", "visit_date"]),
        'as': pd.read_csv('data/air_store_info.csv').set_index("air_store_id"),
        # 'hs': pd.read_csv('data/hpg_store_info.csv'),
        'ar': pd.read_csv(
            'data/air_reserve.csv',
            parse_dates=["visit_datetime", "reserve_datetime"]
        ),
        'hr': pd.read_csv(
            'data/hpg_reserve.csv',
            parse_dates=["visit_datetime", "reserve_datetime"]
        ),
        'id': pd.read_csv('data/store_id_relation.csv'),
        'tes': pd.read_csv('data/sample_submission.csv'),
        'hol': pd.read_csv(
            'data/date_info.csv', parse_dates=["calendar_date"]
        ).rename(columns={'calendar_date': 'visit_date'}).set_index("visit_date"),
        'air_nearest': pd.read_csv('data/air_store_info_with_nearest_active_station.csv'),
    }
    weather_dir = 'data/1-1-16_5-31-17_Weather/'

    data["as"]["prefecture"] = data["as"]["air_area_name"].apply(
        lambda x: x.split(" ")[0])
    data["as"]["municipal"] = data["as"]["air_area_name"].apply(
        lambda x: x.split(" ")[1])
    # Holidays
    holidays = data["hol"][["holiday_flg"]].transpose()
    # Unset holiday flag for weekends
    for col in holidays.columns:
        if col.weekday() == 5 or col.weekday() == 6:
            holidays[col] = 0
    # Visitors
    visitors = data["tra"]["visitors"].unstack(-1, fill_value=0)
    # Reservations
    data["ar"]["source"] = "ar"
    hpg_reserves = data["hr"].merge(data["id"], on="hpg_store_id").drop(
        "hpg_store_id", axis=1)
    hpg_reserves = hpg_reserves[~hpg_reserves["air_store_id"].isnull()]
    # print(hpg_reserves.shape)
    hpg_reserves["source"] = "hpg"
    reserves = pd.concat([data["ar"], hpg_reserves], axis=0)
    print(reserves.isnull().sum().sum())
    print(reserves[reserves.source == "hpg"].shape)
    reserves["reserve_date"] = reserves["reserve_datetime"].dt.date
    reserves["visit_date"] = reserves["visit_datetime"].dt.date
    # reserves["reserve_diff"] = (
    #     reserves["visit_datetime"] - reserves["reserve_datetime"]
    # ).astype("timedelta64[D]").astype("int16")
    reserves = reserves.groupby([
        "air_store_id", "visit_date", "reserve_date", "source"
    ], as_index=False)[["reserve_visitors"]].sum()
    # reserves = reserves.unstack(-1, fill_value=0)
    # Stores
    stores = data["as"].copy()
    stores["genre"] = preprocessing.LabelEncoder(
    ).fit_transform(stores["air_genre_name"])
    stores["area"] = preprocessing.LabelEncoder(
    ).fit_transform(stores["air_area_name"])
    stores["prefecture"] = preprocessing.LabelEncoder(
    ).fit_transform(stores["prefecture"])
    stores["municipal"] = preprocessing.LabelEncoder(
    ).fit_transform(stores["municipal"])
    stores["id"] = preprocessing.LabelEncoder(
    ).fit_transform(stores.index.tolist())

    # Retain only stores that matters
    target_stores = pd.Series(data["tes"]["id"].apply(
        lambda x: "_".join(x.split("_")[:-1]))).unique()
    visitors = visitors.reindex(target_stores)
    stores = stores[["genre", "area", "prefecture", "municipal", "id"]
                    ].reindex(target_stores)

    # Weather
    tmp = []
    for air_id in stores.index.tolist():
        station = data["air_nearest"][
            data["air_nearest"].air_store_id == air_id].station_id.iloc[0]
        weather_data = pd.read_csv(
            weather_dir + station + '.csv', parse_dates=['calendar_date']
        ).rename(columns={'calendar_date': 'visit_date'})[
            ['visit_date', 'precipitation', 'avg_temperature']]
        weather_data["air_store_id"] = air_id
        tmp.append(weather_data)
    df_weather = pd.concat(tmp, axis=0).reset_index(
        drop=True).set_index(["air_store_id", "visit_date"])
    print(visitors.columns)
    precipitation = df_weather[
        "precipitation"].unstack(-1).reindex(
            target_stores)
    avg_temperature = df_weather[
        "avg_temperature"].unstack(-1).reindex(
            target_stores)
    # print(precipitation)
    # print(precipitation.isnull().sum(axis=1))
    return visitors, reserves, stores, holidays, (precipitation, avg_temperature)


def get_timespan(df, dd, delta, periods):
    return df[
        pd.date_range(dd - timedelta(days=delta), periods=periods)
    ].copy()


def preprocess():
    (visitors, reserves, stores, holidays,
     (precipitation, avg_temperature)) = read_data()
    reserves_from_ar = reserves[reserves.source == "ar"].groupby(
        ["air_store_id", "reserve_date"]
    )["reserve_visitors"].sum().unstack(
        -1, fill_value=0
    ).reindex(visitors.index, columns=visitors.columns, fill_value=0)
    reserves_from_hpg = reserves[reserves.source == "hpg"].groupby(
        ["air_store_id", "reserve_date"]
    )["reserve_visitors"].sum().unstack(
        -1, fill_value=0
    ).reindex(visitors.index, columns=visitors.columns, fill_value=0)

    def prepare_dataset(ty2, is_train=True):
        holidays_year2 = get_timespan(
            holidays, ty2, LOOKBACK - 1, LOOKBACK + 38)
        visitors_year2 = visitors.reindex(
            columns=[
                x - timedelta(days=1)
                for x in holidays_year2.columns
            ],
            fill_value=0
        )
        precipitation_year2 = precipitation.reindex(
            columns=holidays_year2.columns)
        avg_temperature_year2 = avg_temperature.reindex(
            columns=holidays_year2.columns)

        # reserves_current_to_ar = reserves[
        #     (reserves.reserve_date < ty2) & (reserves.source == "ar")
        # ].groupby(
        #     ["air_store_id", "visit_date"]
        # )["reserve_visitors"].sum().unstack(-1).reindex(
        #     visitors.index, columns=holidays_year2.columns, fill_value=0)
        # reserves_current_to_hpg = reserves[
        #     (reserves.reserve_date < ty2) & (reserves.source == "hpg")
        # ].groupby(
        #     ["air_store_id", "visit_date"]
        # )["reserve_visitors"].sum().unstack(-1).reindex(
        #     visitors.index, columns=holidays_year2.columns, fill_value=0)
        reserves_current_to = reserves[
            (reserves.reserve_date < ty2)
        ].groupby(
            ["air_store_id", "visit_date"]
        )["reserve_visitors"].sum().unstack(-1, fill_value=0).reindex(
            visitors.index, columns=holidays_year2.columns, fill_value=0)
        # reserves_to_ar_year2 = get_timespan(
        #     reserves_current_to_ar, ty2, LOOKBACK - 1, LOOKBACK + 38)
        # reserves_to_hpg_year2 = get_timespan(
        #     reserves_current_to_hpg, ty2, LOOKBACK - 1, LOOKBACK + 38)
        reserves_to_year2 = get_timespan(
            reserves_current_to, ty2, LOOKBACK - 1, LOOKBACK + 38)
        reserves_from_ar_year2 = get_timespan(
            reserves_from_ar, ty2, LOOKBACK, LOOKBACK).reindex(
                columns=visitors_year2.columns, fill_value=0)
        reserves_from_hpg_year2 = get_timespan(
            reserves_from_hpg, ty2, LOOKBACK, LOOKBACK).reindex(
                columns=visitors_year2.columns, fill_value=0)
        none_zero = (
            (visitors_year2.iloc[:, -66:-38].sum(axis=1).values > 0)
        )
        if is_train:
            none_zero = (
                none_zero & (
                    visitors_year2.iloc[:, -38:].sum(axis=1).values > 0)
            )
        x = np.concatenate(
            [
                np.expand_dims(df.values[none_zero, :], 2)
                for df in (
                    visitors_year2, np.log1p(reserves_from_ar_year2 +
                                             reserves_from_hpg_year2),
                    np.log1p(reserves_to_year2),
                    precipitation_year2.fillna(0),
                    avg_temperature_year2.fillna(
                        method="ffill", axis=1).fillna(
                        method="bfill", axis=1).fillna(0)
                )
            ], axis=2
        ).astype("float64")
        x_int = np.concatenate(
            [
                np.repeat(
                    date_feature[np.newaxis, :, :],
                    np.sum(none_zero), axis=0)
                for date_feature in (
                    # Holiday
                    holidays_year2.values.transpose(),
                    # Day of Week
                    np.array(
                        [[x.weekday() for x in holidays_year2.columns]]).transpose(),
                    # Day of Month
                    np.array(
                        [[x.day - 1 for x in holidays_year2.columns]]).transpose(),
                    # Month
                    np.array(
                        [[x.month - 1 for x in holidays_year2.columns]]).transpose()
                )
            ] +
            [
                np.repeat(
                    store_feature.values[none_zero, np.newaxis, :],
                    LOOKBACK + 38, axis=1)
                for store_feature in (
                    stores[["genre"]], stores[["prefecture"]],
                    stores[["area"]], stores[["municipal"]]
                )
            ] + [
                # Is Zero
                (visitors_year2 == 0).values.astype(
                    "int16")[none_zero, :, np.newaxis],
                # Precipitation NA
                precipitation_year2.isnull().values.astype(
                    "int16")[none_zero, :, np.newaxis],
                # Temperature NA
                avg_temperature_year2.isnull().values.astype(
                    "int16")[none_zero, :, np.newaxis],
            ], axis=2
        ).astype("int16")
        if is_train:
            y = get_timespan(visitors, ty2, 0, 39).values.astype("float64")
            x[:, -38:, 0] = y[none_zero, :38]
            return x, x_int, y[none_zero, :], pd.Series(stores.index.tolist()).loc[none_zero]
        return x, x_int, pd.Series(stores.index.tolist()).loc[none_zero]

    print(stores["id"].describe())
    print("Preparing dataset...")
    ty2 = date(2017, 1, 29)
    assert ty2.weekday() == 6
    x_tmp, x_i_tmp, y_tmp, _ = prepare_dataset(ty2)
    x = np.memmap("cache/xtrain_seq.npy", mode="w+", order="C", dtype="float64",
                  shape=(x_tmp.shape[0], x_tmp.shape[1], x_tmp.shape[2]))
    x_i = np.memmap("cache/xtrain_i_seq.npy", mode="w+", order="C", dtype="int16",
                    shape=(x_tmp.shape[0], x_i_tmp.shape[1], x_i_tmp.shape[2]))
    y = np.memmap("cache/ytrain_seq.npy", mode="w+", order="C", dtype="float64",
                  shape=(x_tmp.shape[0], y_tmp.shape[1]))
    x[:, :, :] = x_tmp
    x_i[:, :, :] = x_i_tmp
    y[:, :] = y_tmp
    x.flush()
    x_i.flush()
    y.flush()
    current_cnt = x_tmp.shape[0]
    for i in range(1, 50):
        delta = timedelta(days=-3 * i)
        x_tmp, x_i_tmp, y_tmp, _ = prepare_dataset(ty2 + delta)
        print(ty2 + delta - timedelta(days=LOOKBACK), ty2 + delta)
        x = np.memmap("cache/xtrain_seq.npy", mode="r+", order="C", dtype="float64",
                      shape=(current_cnt + x_tmp.shape[0], x_tmp.shape[1], x_tmp.shape[2]))
        x_i = np.memmap("cache/xtrain_i_seq.npy", mode="r+", order="C", dtype="int16",
                        shape=(current_cnt + x_tmp.shape[0], x_i_tmp.shape[1], x_i_tmp.shape[2]))
        y = np.memmap("cache/ytrain_seq.npy", mode="r+", order="C", dtype="float64",
                      shape=(current_cnt + x_tmp.shape[0], y_tmp.shape[1]))
        x[current_cnt:, :, :] = x_tmp
        x_i[current_cnt:, :, :] = x_i_tmp
        y[current_cnt:, :] = y_tmp
        x.flush()
        x_i.flush()
        y.flush()
        current_cnt += x_tmp.shape[0]
    print(x.dtype, x_i.dtype, y.dtype)
    print(x.shape, x_i.shape, y.shape)

    ty2 = date(2017, 3, 12)
    assert ty2.weekday() == 6
    x_val, x_i_val, y_val, store_ids = prepare_dataset(ty2)
    x = np.memmap("cache/xval_seq.npy", mode="w+", order="C", dtype="float64",
                  shape=(x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_i = np.memmap("cache/xval_i_seq.npy", mode="w+", order="C", dtype="int16",
                    shape=(x_i_val.shape[0], x_i_val.shape[1], x_i_val.shape[2]))
    y = np.memmap("cache/yval_seq.npy", mode="w+", order="C", dtype="float64",
                  shape=(y_val.shape[0], y_val.shape[1]))
    store_ids.to_pickle("cache/val_seq_id.pkl")
    x[:, :, :] = x_val
    x_i[:, :, :] = x_i_val
    y[:, :] = y_val
    x.flush()
    x_i.flush()
    y.flush()
    print(x_val.shape)

    ty2 = date(2017, 4, 23)
    assert ty2.weekday() == 6
    x_test, x_i_test, test_id = prepare_dataset(ty2, is_train=False)
    x = np.memmap("cache/xtest_seq.npy", mode="w+", order="C", dtype="float64",
                  shape=(x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    x_i = np.memmap("cache/xtest_i_seq.npy", mode="w+", order="C", dtype="int16",
                    shape=(x_i_test.shape[0], x_i_test.shape[1], x_i_test.shape[2]))
    x[:, :, :] = x_test
    x_i[:, :, :] = x_i_test
    x.flush()
    x_i.flush()
    print(x_test.shape)
    test_id.to_pickle("cache/test_id.pkl")
    print(test_id.head())


if __name__ == "__main__":
    preprocess()
