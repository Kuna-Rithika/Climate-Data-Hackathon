# Climate Disease Stock Project

End-to-end local project for forecasting climate-driven dengue outbreak risk and generating pharmaceutical stock signals.

## Project Structure

```text
climate_disease_stock_project/
|-- data/
|   |-- raw/
|   |   |-- dengue_features_train.csv
|   |   `-- dengue_labels_train.csv
|-- models/
|-- main.py
|-- requirements.txt
|-- train.py
`-- README.md
```

## Dataset

Use a Kaggle mirror of the canonical DengAI dataset:

- https://www.kaggle.com/datasets/dipayancodes/dengue
- https://www.kaggle.com/datasets/mzhasan00/dengue/data

Preferred files to place in `data/raw/`:

- `dengue_features_train.csv`
- `dengue_labels_train.csv`

The loader also supports a single combined CSV if your Kaggle mirror packages the data in one file, as long as it contains:

- `city`
- `year`
- `weekofyear`
- `total_cases`
- the weather feature columns used in `train.py`

## Supported Live Prediction Locations

The live inference pipeline is trained for the two cities present in the dataset:

- `sj` or `san_juan`
- `iq` or `iquitos`

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python train.py
```

Run live prediction:

```bash
python main.py --city sj --api-key YOUR_OPENWEATHER_API_KEY
```

Or set an environment variable first:

```bash
set OPENWEATHER_API_KEY=YOUR_OPENWEATHER_API_KEY
python main.py --city iq
```

## Saved Artifacts

After training, the project writes:

- `models/outbreak_model.pkl`
- `models/weather_lstm.h5`
- `models/weather_scaler.pkl`
- `models/metadata.pkl`

## Notes

- `train.py` handles preprocessing, feature engineering, outbreak-model training, LSTM weather training, and artifact saving.
- `main.py` fetches live weather from OpenWeather, forecasts future weekly weather with the saved LSTM, predicts dengue outbreak probability, and generates stock signals.
- The live stock signals use outbreak intensity plus optional `yfinance` momentum data when available.
