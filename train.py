from __future__ import annotations

import math
import os
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

try:
    from xgboost import XGBClassifier

    CLASSIFIER_BACKEND = "xgboost"
except Exception:
    from sklearn.ensemble import RandomForestClassifier

    CLASSIFIER_BACKEND = "random_forest"


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES_FILE = "dengue_features_train.csv"
LABELS_FILE = "dengue_labels_train.csv"

WEATHER_FEATURES = [
    "station_avg_temp_c",
    "station_min_temp_c",
    "station_max_temp_c",
    "station_diur_temp_rng_c",
    "reanalysis_relative_humidity_percent",
    "reanalysis_specific_humidity_g_per_kg",
    "precipitation_amt_mm",
    "station_precip_mm",
    "reanalysis_precip_amt_kg_per_m2",
    "reanalysis_dew_point_temp_k",
]

WINDOW_SIZE = 4
LOOKBACK = 12
VALIDATION_FRACTION = 0.2
OUTBREAK_QUANTILE = 0.75
RANDOM_STATE = 42

SUPPORTED_CITIES = {
    "sj": {
        "display_name": "San Juan, Puerto Rico",
        "lat": 18.4655,
        "lon": -66.1057,
    },
    "iq": {
        "display_name": "Iquitos, Peru",
        "lat": -3.7437,
        "lon": -73.2516,
    },
}


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    return frame


def safe_iso_week_start(year: int, week: int) -> pd.Timestamp:
    week = max(1, min(int(week), 53))
    while week >= 1:
        try:
            return pd.Timestamp(date.fromisocalendar(int(year), int(week), 1))
        except ValueError:
            week -= 1
    return pd.Timestamp(f"{int(year)}-01-01")


def load_training_data() -> pd.DataFrame:
    features_path = DATA_DIR / FEATURES_FILE
    labels_path = DATA_DIR / LABELS_FILE

    if features_path.exists() and labels_path.exists():
        features = normalize_columns(pd.read_csv(features_path))
        labels = normalize_columns(pd.read_csv(labels_path))
        frame = features.merge(labels, on=["city", "year", "weekofyear"], how="inner")
        return frame

    for csv_path in DATA_DIR.glob("*.csv"):
        probe = normalize_columns(pd.read_csv(csv_path, nrows=20))
        if {"city", "year", "weekofyear", "total_cases"}.issubset(probe.columns):
            return normalize_columns(pd.read_csv(csv_path))

    expected = (
        f"Expected either '{FEATURES_FILE}' + '{LABELS_FILE}' in {DATA_DIR} "
        "or a single combined CSV with city/year/weekofyear/total_cases columns."
    )
    raise FileNotFoundError(expected)


def preprocess_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_columns(frame)

    required_columns = {"city", "year", "weekofyear", "total_cases"}
    missing_required = required_columns - set(frame.columns)
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_required)}")

    missing_weather = [col for col in WEATHER_FEATURES if col not in frame.columns]
    if missing_weather:
        raise ValueError(
            "Dataset is missing weather columns required by the project: "
            + ", ".join(missing_weather)
        )

    frame["city"] = frame["city"].astype(str).str.strip().str.lower()
    frame = frame[frame["city"].isin(SUPPORTED_CITIES)].copy()
    if frame.empty:
        raise ValueError("No supported cities were found. Expected rows for 'sj' and/or 'iq'.")

    if "week_start_date" in frame.columns:
        frame["week_start_date"] = pd.to_datetime(frame["week_start_date"], errors="coerce")
    else:
        frame["week_start_date"] = [
            safe_iso_week_start(year, week)
            for year, week in zip(frame["year"].astype(int), frame["weekofyear"].astype(int))
        ]

    frame["weekofyear"] = frame["weekofyear"].astype(int)
    frame["year"] = frame["year"].astype(int)
    frame["total_cases"] = pd.to_numeric(frame["total_cases"], errors="coerce")

    for column in WEATHER_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values(["city", "week_start_date", "year", "weekofyear"]).reset_index(drop=True)

    for _, city_idx in frame.groupby("city").groups.items():
        city_slice = frame.loc[city_idx, WEATHER_FEATURES].copy()
        city_slice = city_slice.interpolate(limit_direction="both")
        city_slice = city_slice.fillna(city_slice.median())
        frame.loc[city_idx, WEATHER_FEATURES] = city_slice

    frame[WEATHER_FEATURES] = frame[WEATHER_FEATURES].fillna(frame[WEATHER_FEATURES].median())
    frame["total_cases"] = frame["total_cases"].fillna(frame["total_cases"].median())

    for column in ["precipitation_amt_mm", "station_precip_mm", "reanalysis_precip_amt_kg_per_m2"]:
        frame[column] = frame[column].clip(lower=0.0)

    frame["station_diur_temp_rng_c"] = frame["station_diur_temp_rng_c"].clip(lower=0.0)
    frame["reanalysis_relative_humidity_percent"] = frame["reanalysis_relative_humidity_percent"].clip(
        lower=0.0,
        upper=100.0,
    )
    return frame


def wrap_week(week_number: int) -> int:
    return ((int(week_number) - 1) % 52) + 1


def engineer_window_features(window: pd.DataFrame, city: str, target_week: int) -> dict[str, float]:
    avg_temp = window["station_avg_temp_c"].to_numpy(dtype=float)
    min_temp = window["station_min_temp_c"].to_numpy(dtype=float)
    max_temp = window["station_max_temp_c"].to_numpy(dtype=float)
    diurnal = window["station_diur_temp_rng_c"].to_numpy(dtype=float)
    humidity = window["reanalysis_relative_humidity_percent"].to_numpy(dtype=float)
    specific_humidity = window["reanalysis_specific_humidity_g_per_kg"].to_numpy(dtype=float)
    precipitation = window["precipitation_amt_mm"].to_numpy(dtype=float)
    station_precip = window["station_precip_mm"].to_numpy(dtype=float)
    reanalysis_precip = window["reanalysis_precip_amt_kg_per_m2"].to_numpy(dtype=float)
    dew_point = window["reanalysis_dew_point_temp_k"].to_numpy(dtype=float)

    avg_temp_mean = float(np.mean(avg_temp))
    humidity_mean = float(np.mean(humidity))
    precip_total = float(np.sum(precipitation))
    station_precip_total = float(np.sum(station_precip))
    reanalysis_precip_total = float(np.sum(reanalysis_precip))

    features = {
        "avg_temp_mean_4w": avg_temp_mean,
        "avg_temp_last": float(avg_temp[-1]),
        "min_temp_low_4w": float(np.min(min_temp)),
        "max_temp_high_4w": float(np.max(max_temp)),
        "diurnal_mean_4w": float(np.mean(diurnal)),
        "humidity_mean_4w": humidity_mean,
        "humidity_last": float(humidity[-1]),
        "specific_humidity_mean_4w": float(np.mean(specific_humidity)),
        "precip_total_4w": precip_total,
        "precip_mean_4w": float(np.mean(precipitation)),
        "station_precip_total_4w": station_precip_total,
        "reanalysis_precip_total_4w": reanalysis_precip_total,
        "dew_point_mean_4w": float(np.mean(dew_point)),
        "temp_trend_4w": float(avg_temp[-1] - avg_temp[0]),
        "humidity_trend_4w": float(humidity[-1] - humidity[0]),
        "precip_volatility_4w": float(np.std(precipitation)),
        "heat_humidity_index": float(avg_temp_mean * (1.0 + (humidity_mean / 100.0))),
        "rain_humidity_interaction": float(precip_total * (humidity_mean / 100.0)),
        "target_week_sin": float(math.sin(2.0 * math.pi * wrap_week(target_week) / 52.0)),
        "target_week_cos": float(math.cos(2.0 * math.pi * wrap_week(target_week) / 52.0)),
        "city_sj": 1.0 if city == "sj" else 0.0,
        "city_iq": 1.0 if city == "iq" else 0.0,
    }
    return features


def build_classifier_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]], list[str]]:
    rows: list[dict[str, Any]] = []
    thresholds: dict[str, dict[str, float]] = {}

    for city, city_frame in frame.groupby("city"):
        city_frame = city_frame.sort_values("week_start_date").reset_index(drop=True)
        outbreak_threshold = float(city_frame["total_cases"].quantile(OUTBREAK_QUANTILE))
        baseline_median = float(city_frame["total_cases"].median())
        severe_threshold = float(city_frame["total_cases"].quantile(0.90))
        thresholds[city] = {
            "baseline_median": baseline_median,
            "outbreak_threshold": outbreak_threshold,
            "severe_threshold": severe_threshold,
        }

        for start_idx in range(0, len(city_frame) - WINDOW_SIZE):
            window = city_frame.iloc[start_idx : start_idx + WINDOW_SIZE]
            target_row = city_frame.iloc[start_idx + WINDOW_SIZE]
            feature_row = engineer_window_features(window, city=city, target_week=int(target_row["weekofyear"]))
            feature_row["target"] = int(float(target_row["total_cases"]) > outbreak_threshold)
            feature_row["target_cases"] = float(target_row["total_cases"])
            feature_row["city"] = city
            feature_row["target_date"] = target_row["week_start_date"]
            rows.append(feature_row)

    classifier_frame = pd.DataFrame(rows)
    classifier_frame = classifier_frame.sort_values(["city", "target_date"]).reset_index(drop=True)
    feature_names = [
        column
        for column in classifier_frame.columns
        if column not in {"target", "target_cases", "city", "target_date"}
    ]
    return classifier_frame, thresholds, feature_names


def time_split_by_city(frame: pd.DataFrame, group_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    valid_parts = []

    for _, group in frame.groupby(group_column):
        group = group.sort_values(group.columns[-1]).reset_index(drop=True)
        split_idx = max(1, int(len(group) * (1.0 - VALIDATION_FRACTION)))
        if split_idx >= len(group):
            split_idx = len(group) - 1
        train_parts.append(group.iloc[:split_idx].copy())
        valid_parts.append(group.iloc[split_idx:].copy())

    train_frame = pd.concat(train_parts, ignore_index=True)
    valid_frame = pd.concat(valid_parts, ignore_index=True)
    return train_frame, valid_frame


def build_classifier_model(scale_pos_weight: float):
    if CLASSIFIER_BACKEND == "xgboost":
        return XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        )

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_classifier(model, x_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_valid, probabilities))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def build_lstm_sequences(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_sequences = []
    y_targets = []
    meta_rows = []

    for city, city_frame in frame.groupby("city"):
        city_frame = city_frame.sort_values("week_start_date").reset_index(drop=True)
        city_values = city_frame[WEATHER_FEATURES].to_numpy(dtype=np.float32)

        for current_idx in range(LOOKBACK, len(city_frame)):
            x_sequences.append(city_values[current_idx - LOOKBACK : current_idx])
            y_targets.append(city_values[current_idx])
            meta_rows.append(
                {
                    "city": city,
                    "target_date": city_frame.loc[current_idx, "week_start_date"],
                    "target_week": int(city_frame.loc[current_idx, "weekofyear"]),
                }
            )

    x_array = np.asarray(x_sequences, dtype=np.float32)
    y_array = np.asarray(y_targets, dtype=np.float32)
    meta_frame = pd.DataFrame(meta_rows)
    return x_array, y_array, meta_frame


def split_lstm_sequences(
    x_sequences: np.ndarray,
    y_targets: np.ndarray,
    meta_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_indices = []
    valid_indices = []

    for _, city_meta in meta_frame.groupby("city"):
        ordered_idx = city_meta.sort_values("target_date").index.to_list()
        split_idx = max(1, int(len(ordered_idx) * (1.0 - VALIDATION_FRACTION)))
        if split_idx >= len(ordered_idx):
            split_idx = len(ordered_idx) - 1
        train_indices.extend(ordered_idx[:split_idx])
        valid_indices.extend(ordered_idx[split_idx:])

    train_indices = np.asarray(train_indices, dtype=int)
    valid_indices = np.asarray(valid_indices, dtype=int)

    return (
        x_sequences[train_indices],
        x_sequences[valid_indices],
        y_targets[train_indices],
        y_targets[valid_indices],
    )


def scale_lstm_data(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, x_train.shape[-1]))

    x_train_scaled = scaler.transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_valid_scaled = scaler.transform(x_valid.reshape(-1, x_valid.shape[-1])).reshape(x_valid.shape)
    y_train_scaled = scaler.transform(y_train)
    y_valid_scaled = scaler.transform(y_valid)

    return x_train_scaled, x_valid_scaled, y_train_scaled, y_valid_scaled, scaler


def build_lstm_model(feature_count: int) -> Sequential:
    model = Sequential(
        [
            LSTM(64, input_shape=(LOOKBACK, feature_count)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(feature_count),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def build_city_medians(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    medians: dict[str, dict[str, float]] = {}
    for city, city_frame in frame.groupby("city"):
        medians[city] = {column: float(city_frame[column].median()) for column in WEATHER_FEATURES}
    return medians


def build_seasonal_profiles(frame: pd.DataFrame) -> dict[str, dict[int, dict[str, float]]]:
    seasonal_profiles: dict[str, dict[int, dict[str, float]]] = {}

    for city, city_frame in frame.groupby("city"):
        profile_frame = city_frame.groupby("weekofyear")[WEATHER_FEATURES].median().reset_index()
        seasonal_profiles[city] = {
            int(row["weekofyear"]): {feature: float(row[feature]) for feature in WEATHER_FEATURES}
            for _, row in profile_frame.iterrows()
        }

    return seasonal_profiles


def build_training_periods(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    periods: dict[str, dict[str, Any]] = {}
    for city, city_frame in frame.groupby("city"):
        periods[city] = {
            "rows": int(len(city_frame)),
            "start": str(city_frame["week_start_date"].min().date()),
            "end": str(city_frame["week_start_date"].max().date()),
        }
    return periods


def train_outbreak_model(classifier_frame: pd.DataFrame, feature_names: list[str]):
    train_frame, valid_frame = time_split_by_city(
        classifier_frame[feature_names + ["target", "city", "target_date"]],
        group_column="city",
    )

    x_train = train_frame[feature_names]
    y_train = train_frame["target"].astype(int)
    x_valid = valid_frame[feature_names]
    y_valid = valid_frame["target"].astype(int)

    positives = max(int(y_train.sum()), 1)
    negatives = max(int(len(y_train) - y_train.sum()), 1)
    scale_pos_weight = negatives / positives

    eval_model = build_classifier_model(scale_pos_weight=scale_pos_weight)
    eval_model.fit(x_train, y_train)
    metrics = evaluate_classifier(eval_model, x_valid, y_valid)

    final_model = build_classifier_model(scale_pos_weight=scale_pos_weight)
    final_model.fit(classifier_frame[feature_names], classifier_frame["target"].astype(int))
    return final_model, metrics


def train_weather_model(frame: pd.DataFrame):
    x_sequences, y_targets, meta_frame = build_lstm_sequences(frame)
    x_train, x_valid, y_train, y_valid = split_lstm_sequences(x_sequences, y_targets, meta_frame)
    x_train_scaled, x_valid_scaled, y_train_scaled, y_valid_scaled, scaler = scale_lstm_data(
        x_train=x_train,
        x_valid=x_valid,
        y_train=y_train,
        y_valid=y_valid,
    )

    model = build_lstm_model(feature_count=len(WEATHER_FEATURES))
    callbacks = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    history = model.fit(
        x_train_scaled,
        y_train_scaled,
        validation_data=(x_valid_scaled, y_valid_scaled),
        epochs=60,
        batch_size=16,
        verbose=0,
        callbacks=callbacks,
    )
    val_loss = float(model.evaluate(x_valid_scaled, y_valid_scaled, verbose=0))
    return model, scaler, {"val_mse": val_loss, "epochs_ran": len(history.history["loss"])}


def save_artifacts(
    outbreak_model,
    weather_model: Sequential,
    weather_scaler: StandardScaler,
    metadata: dict[str, Any],
) -> None:
    joblib.dump(outbreak_model, MODELS_DIR / "outbreak_model.pkl")
    weather_model.save(MODELS_DIR / "weather_lstm.h5")
    joblib.dump(weather_scaler, MODELS_DIR / "weather_scaler.pkl")
    joblib.dump(metadata, MODELS_DIR / "metadata.pkl")


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_directories()
    set_global_seed()

    frame = preprocess_frame(load_training_data())
    classifier_frame, thresholds, feature_names = build_classifier_dataset(frame)

    outbreak_model, classifier_metrics = train_outbreak_model(
        classifier_frame=classifier_frame,
        feature_names=feature_names,
    )
    weather_model, weather_scaler, weather_metrics = train_weather_model(frame)

    metadata = {
        "classifier_backend": CLASSIFIER_BACKEND,
        "feature_names": feature_names,
        "weather_features": WEATHER_FEATURES,
        "window_size": WINDOW_SIZE,
        "lookback": LOOKBACK,
        "forecast_steps": WINDOW_SIZE - 1,
        "supported_cities": SUPPORTED_CITIES,
        "outbreak_thresholds": thresholds,
        "city_feature_medians": build_city_medians(frame),
        "seasonal_weather_profiles": build_seasonal_profiles(frame),
        "training_periods": build_training_periods(frame),
        "classifier_metrics": classifier_metrics,
        "weather_metrics": weather_metrics,
    }

    save_artifacts(
        outbreak_model=outbreak_model,
        weather_model=weather_model,
        weather_scaler=weather_scaler,
        metadata=metadata,
    )

    print("Training complete.")
    print(f"Dataset rows: {len(frame)}")
    print(f"Cities: {sorted(frame['city'].unique().tolist())}")
    print(f"Classifier backend: {CLASSIFIER_BACKEND}")
    print(f"Classifier metrics: {classifier_metrics}")
    print(f"Weather metrics: {weather_metrics}")
    print(f"Saved outbreak model to: {MODELS_DIR / 'outbreak_model.pkl'}")
    print(f"Saved LSTM model to: {MODELS_DIR / 'weather_lstm.h5'}")
    print(f"Saved scaler to: {MODELS_DIR / 'weather_scaler.pkl'}")
    print(f"Saved metadata to: {MODELS_DIR / 'metadata.pkl'}")


if __name__ == "__main__":
    main()
