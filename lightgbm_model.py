"""The LightGBM model from Colab (lightgbm_bus_delay.joblib), shown as an extra prediction in the game.

It only uses the timetable: stop position, hour, route, direction and weekday, no live data.

    python lightgbm_model.py eval   # score it on the last logged day against delay_model / enroute_model
"""
import sys

import joblib
import numpy as np
import pandas as pd

MODEL_FILE = "lightgbm_bus_delay.joblib"
FEATURES = ["stop_position", "hour", "route", "direction", "weekday"]


def frame(route, trip_id, stop_position, sched_sec, categories):
    """Model inputs. Direction and weekday come from the trip id: 'HSL:1055_20261023_Su_1_0648'."""
    parts = pd.Series(trip_id).str.split("_", expand=True)
    df = pd.DataFrame({"stop_position": stop_position, "hour": np.asarray(sched_sec) // 3600,
                       "route": route, "direction": parts[3].to_numpy(), "weekday": parts[2].to_numpy()})
    for col, cats in zip(["route", "direction", "weekday"], categories):
        df[col] = pd.Categorical(df[col], categories=cats)  # unknown values become missing
    return df[FEATURES]


class LightGBMModel:
    def __init__(self, path=MODEL_FILE):
        self.model = joblib.load(path)["model"]
        self.categories = self.model.booster_.pandas_categorical

    def predict_many(self, route, trip_id, stop_position, sched_sec):
        return self.model.predict(frame(route, trip_id, stop_position, sched_sec, self.categories))

    def predict(self, route, trip_id, stop_position, sched_sec):
        return float(self.predict_many([route], [trip_id], [stop_position], [sched_sec])[0])


def evaluate():
    import delay_model
    import enroute_model

    lgb = LightGBMModel()

    # 1. Normal guesses: the same rows and lead times as `python delay_model.py eval`.
    df, rng = delay_model.prepare()
    days = sorted(df["date"].unique())
    train, test = df[df["date"] != days[-1]], df[df["date"] == days[-1]]
    general = delay_model.fit(train, rng)
    test = delay_model.apply_encodings(test, delay_model.encoding_tables(train)).reset_index(drop=True)
    y = test["delay"].to_numpy()
    preds = {"general model": general.predict(test[delay_model.FEATURES]),
             "LightGBM": lgb.predict_many(test["route"], test["trip_id"], test["stop_position"], test["sched"]),
             "always 0": np.zeros(len(y))}
    print(f"\n1. Normal guesses, {days[-1]}, {len(y):,} departures (MAE, seconds):")
    for name, p in preds.items():
        print(f"  {name:16s} {np.mean(np.abs(p - y)):6.1f}")

    # 2. Bus already running 1-10 stops before the stop: the notebook's / enroute_model's test pairs.
    en = enroute_model.load()
    sums = enroute_model.hist_sums(en[en["date"] != days[-1]])
    pairs = enroute_model.build_pairs(en[en["date"] == days[-1]], enroute_model.hist_means(sums))
    y = pairs["delay_target"].to_numpy()
    p = lgb.predict_many(pairs["route"], pairs["trip_id"], pairs["pos_target"], pairs["sched_target"])
    print(f"\n2. Bus 1-10 stops away, {days[-1]}, {len(y):,} pairs (MAE, seconds):")
    print(f"  LightGBM         {np.mean(np.abs(p - y)):6.1f}")
    print(f"  copy the delay   {np.mean(np.abs(pairs['delay_now'].to_numpy() - y)):6.1f}")
    print("  en-route model     32.0  (from `python enroute_model.py eval`)")


if __name__ == "__main__":
    {"eval": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "eval"]()
