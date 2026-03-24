from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model

try:
    import yfinance as yf
except Exception:
    yf = None


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FIVE_DAY_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

STOCK_UNIVERSE = [
    {
        "ticker": "GSK",
        "name": "GSK plc",
        "exposure": 0.95,
        "thesis": "vaccines and infectious-disease portfolio",
    },
    {
        "ticker": "MRK",
        "name": "Merck & Co.",
        "exposure": 0.85,
        "thesis": "anti-infective exposure and emerging-market demand",
    },
    {
        "ticker": "PFE",
        "name": "Pfizer",
        "exposure": 0.65,
        "thesis": "broad pharmaceutical demand response",
    },
    {
        "ticker": "JNJ",
        "name": "Johnson & Johnson",
        "exposure": 0.60,
        "thesis": "defensive healthcare allocation and broad medical demand",
    },
    {
        "ticker": "ABBV",
        "name": "AbbVie",
        "exposure": 0.50,
        "thesis": "diversified biopharma hedge",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live climate-driven dengue outbreak forecasting.")
    parser.add_argument(
        "--city",
        default="sj",
        help="Supported values: sj, san_juan, iq, iquitos",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENWEATHER_API_KEY"),
        help="OpenWeather API key. If omitted, OPENWEATHER_API_KEY env var is used.",
    )
    return parser.parse_args()


def normalize_city(user_value: str) -> str:
    normalized = str(user_value).strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "sj": "sj",
        "san_juan": "sj",
        "sanjuan": "sj",
        "iq": "iq",
        "iquitos": "iq",
    }
    if normalized not in alias_map:
        raise ValueError("Unsupported city. Use one of: sj, san_juan, iq, iquitos.")
    return alias_map[normalized]


def wrap_week(week_number: int) -> int:
    return ((int(week_number) - 1) % 52) + 1


def compute_dew_point_c(temp_c: float, humidity_pct: float) -> float:
    humidity_pct = float(np.clip(humidity_pct, 1.0, 100.0))
    gamma = math.log(humidity_pct / 100.0) + (17.625 * temp_c) / (243.04 + temp_c)
    return (243.04 * gamma) / (17.625 - gamma)


def compute_specific_humidity_g_per_kg(temp_c: float, humidity_pct: float, pressure_hpa: float) -> float:
    humidity_pct = float(np.clip(humidity_pct, 1.0, 100.0))
    pressure_hpa = max(float(pressure_hpa), 300.0)
    saturation_vapor_pressure = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
    vapor_pressure = (humidity_pct / 100.0) * saturation_vapor_pressure
    specific_humidity = 622.0 * vapor_pressure / (pressure_hpa - (0.378 * vapor_pressure))
    return float(max(specific_humidity, 0.0))


def engineer_window_features(window_rows: list[dict[str, float]], city: str, target_week: int) -> dict[str, float]:
    def values(feature_name: str) -> np.ndarray:
        return np.asarray([row[feature_name] for row in window_rows], dtype=float)

    avg_temp = values("station_avg_temp_c")
    min_temp = values("station_min_temp_c")
    max_temp = values("station_max_temp_c")
    diurnal = values("station_diur_temp_rng_c")
    humidity = values("reanalysis_relative_humidity_percent")
    specific_humidity = values("reanalysis_specific_humidity_g_per_kg")
    precipitation = values("precipitation_amt_mm")
    station_precip = values("station_precip_mm")
    reanalysis_precip = values("reanalysis_precip_amt_kg_per_m2")
    dew_point = values("reanalysis_dew_point_temp_k")

    avg_temp_mean = float(np.mean(avg_temp))
    humidity_mean = float(np.mean(humidity))
    precip_total = float(np.sum(precipitation))

    return {
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
        "station_precip_total_4w": float(np.sum(station_precip)),
        "reanalysis_precip_total_4w": float(np.sum(reanalysis_precip)),
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


def load_artifacts() -> tuple[Any, Any, Any, dict[str, Any]]:
    required_paths = {
        "outbreak_model": MODELS_DIR / "outbreak_model.pkl",
        "weather_model": MODELS_DIR / "weather_lstm.h5",
        "weather_scaler": MODELS_DIR / "weather_scaler.pkl",
        "metadata": MODELS_DIR / "metadata.pkl",
    }

    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trained artifacts: " + ", ".join(missing) + ". Run train.py first."
        )

    outbreak_model = joblib.load(required_paths["outbreak_model"])
    weather_model = load_model(required_paths["weather_model"])
    weather_scaler = joblib.load(required_paths["weather_scaler"])
    metadata = joblib.load(required_paths["metadata"])
    return outbreak_model, weather_model, weather_scaler, metadata


def fetch_openweather_data(api_key: str, city_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not api_key:
        raise ValueError(
            "OpenWeather API key is required. Pass --api-key or set OPENWEATHER_API_KEY."
        )

    params = {
        "lat": city_cfg["lat"],
        "lon": city_cfg["lon"],
        "appid": api_key,
        "units": "metric",
    }

    current_response = requests.get(CURRENT_WEATHER_URL, params=params, timeout=20)
    current_response.raise_for_status()

    forecast_response = requests.get(FIVE_DAY_FORECAST_URL, params=params, timeout=20)
    forecast_response.raise_for_status()

    return current_response.json(), forecast_response.json()


def extract_forecast_rain_mm(forecast_payload: dict[str, Any]) -> float:
    rain_total = 0.0
    for item in forecast_payload.get("list", []):
        rain_total += float(item.get("rain", {}).get("3h", 0.0))
    return rain_total


def sanitize_weather_vector(vector: dict[str, float], city_medians: dict[str, float]) -> dict[str, float]:
    sanitized = dict(city_medians)
    sanitized.update(vector)

    sanitized["reanalysis_relative_humidity_percent"] = float(
        np.clip(sanitized["reanalysis_relative_humidity_percent"], 0.0, 100.0)
    )
    for column in ["precipitation_amt_mm", "station_precip_mm", "reanalysis_precip_amt_kg_per_m2"]:
        sanitized[column] = float(max(sanitized[column], 0.0))

    sanitized["station_diur_temp_rng_c"] = float(max(sanitized["station_diur_temp_rng_c"], 0.0))

    if sanitized["station_max_temp_c"] < sanitized["station_min_temp_c"]:
        sanitized["station_min_temp_c"], sanitized["station_max_temp_c"] = (
            sanitized["station_max_temp_c"],
            sanitized["station_min_temp_c"],
        )

    sanitized["station_avg_temp_c"] = float(
        np.clip(
            sanitized["station_avg_temp_c"],
            sanitized["station_min_temp_c"],
            sanitized["station_max_temp_c"],
        )
    )
    return sanitized


def build_live_weather_vector(
    current_payload: dict[str, Any],
    forecast_payload: dict[str, Any],
    city_medians: dict[str, float],
) -> dict[str, float]:
    current_main = current_payload.get("main", {})
    forecast_items = forecast_payload.get("list", [])

    temp_values = [float(current_main.get("temp", city_medians["station_avg_temp_c"]))]
    humidity_values = [
        float(current_main.get("humidity", city_medians["reanalysis_relative_humidity_percent"]))
    ]
    pressure_values = [float(current_main.get("pressure", 1013.25))]
    min_candidates = [float(current_main.get("temp_min", temp_values[0]))]
    max_candidates = [float(current_main.get("temp_max", temp_values[0]))]

    for item in forecast_items:
        item_main = item.get("main", {})
        temp_values.append(float(item_main.get("temp", temp_values[-1])))
        humidity_values.append(float(item_main.get("humidity", humidity_values[-1])))
        pressure_values.append(float(item_main.get("pressure", pressure_values[-1])))
        min_candidates.append(float(item_main.get("temp_min", min_candidates[-1])))
        max_candidates.append(float(item_main.get("temp_max", max_candidates[-1])))

    avg_temp_c = float(np.mean(temp_values))
    humidity_pct = float(np.mean(humidity_values))
    pressure_hpa = float(np.mean(pressure_values))
    min_temp_c = float(np.min(min_candidates))
    max_temp_c = float(np.max(max_candidates))
    rain_mm_5d = extract_forecast_rain_mm(forecast_payload)
    estimated_weekly_rain = float(max(rain_mm_5d * (7.0 / 5.0), 0.0))
    dew_point_c = compute_dew_point_c(avg_temp_c, humidity_pct)
    specific_humidity = compute_specific_humidity_g_per_kg(avg_temp_c, humidity_pct, pressure_hpa)

    vector = dict(city_medians)
    vector.update(
        {
            "station_avg_temp_c": avg_temp_c,
            "station_min_temp_c": min_temp_c,
            "station_max_temp_c": max_temp_c,
            "station_diur_temp_rng_c": float(max(max_temp_c - min_temp_c, 0.0)),
            "reanalysis_relative_humidity_percent": float(np.clip(humidity_pct, 0.0, 100.0)),
            "reanalysis_specific_humidity_g_per_kg": specific_humidity,
            "precipitation_amt_mm": estimated_weekly_rain,
            "station_precip_mm": estimated_weekly_rain,
            "reanalysis_precip_amt_kg_per_m2": estimated_weekly_rain,
            "reanalysis_dew_point_temp_k": dew_point_c + 273.15,
        }
    )
    return sanitize_weather_vector(vector, city_medians)


def get_seasonal_profile(
    metadata: dict[str, Any],
    city: str,
    week_number: int,
) -> dict[str, float]:
    weather_features = metadata["weather_features"]
    city_medians = metadata["city_feature_medians"][city]
    city_profiles = metadata["seasonal_weather_profiles"].get(city, {})
    raw_profile = city_profiles.get(wrap_week(week_number), city_medians)
    return {feature: float(raw_profile.get(feature, city_medians[feature])) for feature in weather_features}


def forecast_future_weather(
    weather_model,
    weather_scaler,
    metadata: dict[str, Any],
    city: str,
    current_vector: dict[str, float],
) -> list[dict[str, float]]:
    weather_features = metadata["weather_features"]
    lookback = int(metadata["lookback"])
    forecast_steps = int(metadata["forecast_steps"])
    city_medians = metadata["city_feature_medians"][city]
    current_week = datetime.utcnow().isocalendar().week

    seed_rows = []
    for offset in range(lookback - 1, 0, -1):
        historical_profile = get_seasonal_profile(metadata, city, current_week - offset)
        seed_rows.append([historical_profile[feature] for feature in weather_features])

    current_profile = get_seasonal_profile(metadata, city, current_week)
    blended_current = {
        feature: (0.8 * current_vector[feature]) + (0.2 * current_profile[feature])
        for feature in weather_features
    }
    blended_current = sanitize_weather_vector(blended_current, city_medians)
    seed_rows.append([blended_current[feature] for feature in weather_features])

    sequence = np.asarray(seed_rows, dtype=np.float32)
    predictions: list[dict[str, float]] = []

    for step in range(1, forecast_steps + 1):
        scaled_sequence = weather_scaler.transform(sequence).reshape(1, lookback, len(weather_features))
        scaled_prediction = weather_model.predict(scaled_sequence, verbose=0)[0]
        prediction = weather_scaler.inverse_transform(scaled_prediction.reshape(1, -1))[0]
        future_week = wrap_week(current_week + step)
        seasonal_profile = get_seasonal_profile(metadata, city, future_week)

        blended_prediction = {
            feature: (0.7 * float(prediction[idx])) + (0.3 * seasonal_profile[feature])
            for idx, feature in enumerate(weather_features)
        }
        blended_prediction = sanitize_weather_vector(blended_prediction, city_medians)
        predictions.append(blended_prediction)

        next_row = np.asarray([blended_prediction[feature] for feature in weather_features], dtype=np.float32)
        sequence = np.vstack([sequence[1:], next_row])

    return predictions


def classify_outbreak_risk(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.55:
        return "Elevated"
    if probability >= 0.35:
        return "Moderate"
    return "Low"


def estimate_case_band(probability: float, threshold_meta: dict[str, float]) -> tuple[int, int]:
    baseline = threshold_meta["baseline_median"]
    severe = threshold_meta["severe_threshold"]
    center = baseline + probability * (severe - baseline)
    lower = max(int(round(center * 0.85)), 0)
    upper = max(int(round(center * 1.15)), lower)
    return lower, upper


def fetch_stock_snapshot(ticker: str) -> dict[str, float] | None:
    if yf is None:
        return None

    try:
        history = yf.download(
            tickers=ticker,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception:
        return None

    if history is None or history.empty or "Close" not in history.columns:
        return None

    closes = history["Close"].dropna()
    if len(closes) < 20:
        return None

    last_price = float(closes.iloc[-1])
    sma20 = float(closes.rolling(20).mean().iloc[-1])
    sma60 = float(closes.rolling(60).mean().iloc[-1]) if len(closes) >= 60 else sma20

    if len(closes) >= 31:
        return_30d = float((closes.iloc[-1] / closes.iloc[-31]) - 1.0)
    else:
        return_30d = 0.0

    momentum_score = 0.5
    momentum_score += 0.20 if sma20 >= sma60 else -0.10
    momentum_score += float(np.clip(return_30d / 0.25, -0.20, 0.20))
    momentum_score = float(np.clip(momentum_score, 0.0, 1.0))

    return {
        "last_price": last_price,
        "return_30d": return_30d,
        "momentum_score": momentum_score,
    }


def build_stock_signals(outbreak_probability: float) -> list[dict[str, Any]]:
    ranked = []

    for stock in STOCK_UNIVERSE:
        snapshot = fetch_stock_snapshot(stock["ticker"])
        momentum_score = snapshot["momentum_score"] if snapshot else 0.5
        signal_score = float(
            np.clip((0.75 * outbreak_probability * stock["exposure"]) + (0.25 * momentum_score), 0.0, 1.0)
        )

        if signal_score >= 0.75:
            recommendation = "BUY"
        elif signal_score >= 0.55:
            recommendation = "WATCH"
        else:
            recommendation = "HOLD"

        ranked.append(
            {
                "ticker": stock["ticker"],
                "name": stock["name"],
                "recommendation": recommendation,
                "signal_score": round(signal_score, 3),
                "reason": stock["thesis"],
                "last_price": round(snapshot["last_price"], 2) if snapshot else None,
                "return_30d_pct": round(snapshot["return_30d"] * 100.0, 2) if snapshot else None,
            }
        )

    ranked.sort(key=lambda item: item["signal_score"], reverse=True)
    return ranked


def pretty_print_results(
    city_cfg: dict[str, Any],
    current_vector: dict[str, float],
    forecast_vectors: list[dict[str, float]],
    outbreak_probability: float,
    threshold_meta: dict[str, float],
    stock_signals: list[dict[str, Any]],
) -> None:
    risk_level = classify_outbreak_risk(outbreak_probability)
    lower_cases, upper_cases = estimate_case_band(outbreak_probability, threshold_meta)

    print("Climate-Driven Disease Outbreak Forecast")
    print("=" * 42)
    print(f"Location: {city_cfg['display_name']}")
    print(f"Timestamp (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current estimated avg temp (C): {current_vector['station_avg_temp_c']:.2f}")
    print(f"Current estimated humidity (%): {current_vector['reanalysis_relative_humidity_percent']:.2f}")
    print(f"Weekly precipitation proxy (mm): {current_vector['precipitation_amt_mm']:.2f}")
    print(f"4-week-ahead dengue outbreak probability: {outbreak_probability:.2%}")
    print(f"Risk level: {risk_level}")
    print(f"Estimated weekly case band: {lower_cases} - {upper_cases}")
    print("")
    print("Predicted Weekly Weather Path")
    print("-" * 29)

    for index, forecast in enumerate(forecast_vectors, start=1):
        print(
            f"Week +{index}: "
            f"temp={forecast['station_avg_temp_c']:.2f}C, "
            f"humidity={forecast['reanalysis_relative_humidity_percent']:.2f}%, "
            f"rain={forecast['precipitation_amt_mm']:.2f}mm"
        )

    print("")
    print("Pharmaceutical Stock Signals")
    print("-" * 30)
    for stock in stock_signals:
        price_text = f"${stock['last_price']:.2f}" if stock["last_price"] is not None else "N/A"
        return_text = (
            f"{stock['return_30d_pct']:+.2f}%"
            if stock["return_30d_pct"] is not None
            else "N/A"
        )
        print(
            f"{stock['ticker']}: {stock['recommendation']} | "
            f"score={stock['signal_score']:.3f} | "
            f"price={price_text} | 30d={return_text} | "
            f"{stock['reason']}"
        )


def main() -> None:
    args = parse_args()
    outbreak_model, weather_model, weather_scaler, metadata = load_artifacts()

    city = normalize_city(args.city)
    city_cfg = metadata["supported_cities"][city]
    city_medians = metadata["city_feature_medians"][city]
    current_payload, forecast_payload = fetch_openweather_data(args.api_key, city_cfg)

    live_vector = build_live_weather_vector(
        current_payload=current_payload,
        forecast_payload=forecast_payload,
        city_medians=city_medians,
    )
    forecast_vectors = forecast_future_weather(
        weather_model=weather_model,
        weather_scaler=weather_scaler,
        metadata=metadata,
        city=city,
        current_vector=live_vector,
    )

    target_week = wrap_week(datetime.utcnow().isocalendar().week + len(forecast_vectors) + 1)
    window_rows = [live_vector] + forecast_vectors
    feature_row = engineer_window_features(window_rows, city=city, target_week=target_week)
    ordered_features = metadata["feature_names"]
    feature_frame = pd.DataFrame(
        [{feature: float(feature_row[feature]) for feature in ordered_features}],
        columns=ordered_features,
    )
    outbreak_probability = float(outbreak_model.predict_proba(feature_frame)[0, 1])

    stock_signals = build_stock_signals(outbreak_probability)
    pretty_print_results(
        city_cfg=city_cfg,
        current_vector=live_vector,
        forecast_vectors=forecast_vectors,
        outbreak_probability=outbreak_probability,
        threshold_meta=metadata["outbreak_thresholds"][city],
        stock_signals=stock_signals,
    )


if __name__ == "__main__":
    main()
