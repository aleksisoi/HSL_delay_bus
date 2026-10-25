"""
python delay_model.py eval    # train on all days but the last, report MAE on the last day
python delay_model.py train   # train on everything and save delay_model.joblib

"""
import os
import sqlite3
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DB = os.environ.get("DELAY_DB", "route_all.db")
MODEL_FILE = "delay_model.joblib"
MAX_LEAD = 2 * 3600        # live features are simulated for guesses made up to 2 h before departure
TRAIN_ROWS = 1_000_000     # rows used to fit the trees (the lookup tables use all rows)

# Historical median delay for these key combinations becomes a feature ("enc_<name>").
ENCODINGS = {
    "route_dir_stop": ["route", "dir", "stop_id"],
    "route_dir_stop_slot": ["route", "dir", "stop_id", "sched"],  # same departure on other days
    "route_dir_hour_wkend": ["route", "dir", "hour", "weekend"],
    "route_hour": ["route", "hour"],
    "stop_hour": ["stop_id", "hour"],
}

FEATURES = (
    ["sched", "dow", "weekend", "stop_position", "into_trip", "dir_num", "lead",
     "stop_last", "stop_last_age", "stop_last3", "trip_last", "trip_last_age", "trip_stops_since"]
    + [f"enc_{n}" for n in ENCODINGS] + [f"cnt_{n}" for n in ENCODINGS]
)


# ---------------------------------------------------------------- data

def parse_trip(trip_ids):
    """'HSL:1506_20261014_To_2_1541' -> direction '2', trip start 15:41 in seconds."""
    parts = trip_ids.str.split("_", expand=True)
    start = pd.to_numeric(parts[4], errors="coerce")
    return parts[3], (start // 100) * 3600 + (start % 100) * 60


def load(db=DATA_DB):
    conn = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True)
    df = pd.read_sql_query(
        "SELECT service_date AS date, route, trip_id, stop_id, stop_position, "
        "scheduled_departure_sec AS sched, delay_seconds AS delay "
        "FROM stoptimes WHERE is_realtime = 1", conn)
    df["dir"], df["trip_start"] = parse_trip(df["trip_id"])
    add_calendar(df)
    return df


def add_calendar(df):
    dow = pd.to_datetime(df["date"], format="%Y%m%d").dt.dayofweek
    df["dow"] = dow
    df["weekend"] = (dow >= 5).astype(int)
    df["hour"] = (df["sched"] // 3600).astype(int)
    df["into_trip"] = df["sched"] - df["trip_start"]
    df["dir_num"] = pd.to_numeric(df["dir"], errors="coerce")


# ---------------------------------------------------------------- live features (simulated for training)

def last_seen(df, keys, now, n=1):
    """For each row: the n most recent departures in the same `keys` group that had
    actually left by `now`. Returns (index arrays into df, validity masks)."""
    group = df.groupby(keys, sort=False, observed=True).ngroup().to_numpy().astype(np.int64)
    left = (df["sched"] + df["delay"]).to_numpy().astype(np.int64)
    order = np.lexsort((left, group))
    sorted_key = group[order] * 1_000_000 + left[order] + 100_000
    pos = np.searchsorted(sorted_key, group * 1_000_000 + now + 100_000, side="right") - 1

    out = []
    for k in range(n):
        p = pos - k
        ok = p >= 0
        p = np.clip(p, 0, None)
        ok &= group[order][p] == group
        out.append((order[p], ok))
    return out


def add_live_features(df, rng):
    """Pretend each row is guessed `lead` seconds before its scheduled departure and
    compute what the server could have seen at that moment."""
    df["lead"] = rng.uniform(60, MAX_LEAD, len(df)).round()
    now = (df["sched"] - df["lead"]).to_numpy().astype(np.int64)

    # Only departures that haven't left yet can be guessed.
    df.drop(df.index[(df["sched"] + df["delay"]).to_numpy() <= now], inplace=True)
    df.reset_index(drop=True, inplace=True)
    now = (df["sched"] - df["lead"]).to_numpy().astype(np.int64)

    delay = df["delay"].to_numpy().astype(float)
    left = (df["sched"] + df["delay"]).to_numpy()
    position = df["stop_position"].to_numpy()

    stop_hits = last_seen(df, ["date", "route", "stop_id"], now, n=3)
    (i0, ok0) = stop_hits[0]
    df["stop_last"] = np.where(ok0, delay[i0], np.nan)
    df["stop_last_age"] = np.where(ok0, now - left[i0], np.nan)
    vals = np.stack([np.where(ok, delay[i], np.nan) for i, ok in stop_hits])
    with np.errstate(all="ignore"):
        df["stop_last3"] = np.nanmean(vals, axis=0)

    ((t0, okt),) = last_seen(df, ["date", "trip_id"], now)
    okt &= position[t0] < position
    df["trip_last"] = np.where(okt, delay[t0], np.nan)
    df["trip_last_age"] = np.where(okt, now - left[t0], np.nan)
    df["trip_stops_since"] = np.where(okt, position - position[t0], np.nan)


# ---------------------------------------------------------------- historical lookup tables

def encoding_tables(src):
    return {name: src.groupby(keys, observed=True)["delay"].agg(enc="median", cnt="size")
            for name, keys in ENCODINGS.items()}


def apply_encodings(df, tables):
    for name, keys in ENCODINGS.items():
        t = tables[name].rename(columns={"enc": f"enc_{name}", "cnt": f"cnt_{name}"})
        df = df.merge(t, left_on=keys, right_index=True, how="left")
    return df


def encode_leave_one_day_out(rows, source):
    """Training rows get lookup values computed from the *other* days, like at prediction time."""
    parts = []
    for day in rows["date"].unique():
        parts.append(apply_encodings(rows[rows["date"] == day],
                                     encoding_tables(source[source["date"] != day])))
    return pd.concat(parts)


# ---------------------------------------------------------------- training

def fit(df, rng):
    rows = df.sample(min(TRAIN_ROWS, len(df)), random_state=int(rng.integers(1 << 31)))
    rows = encode_leave_one_day_out(rows, df)
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=500, learning_rate=0.08,
                                          max_leaf_nodes=127, min_samples_leaf=100,
                                          early_stopping=True, random_state=0)
    model.fit(rows[FEATURES], rows["delay"].clip(-600, 1800))
    return model


def prepare(df=None, seed=0):
    rng = np.random.default_rng(seed)
    df = load() if df is None else df
    add_live_features(df, rng)
    return df, rng


def evaluate():
    t0 = time.time()
    df, rng = prepare()
    days = sorted(df["date"].unique())
    train, test = df[df["date"] != days[-1]], df[df["date"] == days[-1]]
    print(f"loaded {len(df):,} rows in {time.time() - t0:.0f}s; train {days[:-1]}, test {days[-1]}")

    model = fit(train, rng)
    test = apply_encodings(test, encoding_tables(train)).reset_index(drop=True)
    pred = model.predict(test[FEATURES])
    y = test["delay"].to_numpy()

    results = {"new model": pred,
               "old model (mixed LM)": old_model_predictions(train, test),
               "always 0": np.zeros(len(y)),
               "route median": test["route"].map(train.groupby("route")["delay"].median())
                               .fillna(train["delay"].median()).to_numpy()}
    print(f"\nMAE on {len(y):,} test departures (seconds):")
    for name, p in results.items():
        print(f"  {name:22s} {np.mean(np.abs(p - y)):6.1f}")

    print("\nnew vs old by lead time:")
    buckets = pd.cut(test["lead"], [0, 300, 900, 1800, 3600, MAX_LEAD],
                     labels=["<5 min", "5-15 min", "15-30 min", "30-60 min", "1-2 h"])
    table = pd.DataFrame({"lead": buckets, "new": np.abs(pred - y),
                          "old": np.abs(results["old model (mixed LM)"] - y)})
    print(table.groupby("lead", observed=True).mean().round(1).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


def old_model_predictions(train, test):
    """The model game_server.py used before, for comparison."""
    import statsmodels.formula.api as smf
    s = train.sample(150_000, random_state=0).copy()
    s["delay_seconds"] = s["delay"].clip(-300, 1800)
    m = smf.mixedlm("delay_seconds ~ C(hour) + weekend + stop_position", s, groups=s["route"]).fit()
    fixed = m.predict(test[["hour", "weekend", "stop_position"]])
    effects = test["route"].map({r: v.iloc[0] for r, v in m.random_effects.items()}).fillna(0)
    return (fixed + effects).to_numpy()


def train_and_save():
    t0 = time.time()
    df, rng = prepare()
    model = fit(df, rng)
    joblib.dump({"model": model, "tables": encoding_tables(df)}, MODEL_FILE, compress=3)
    print(f"saved {MODEL_FILE} ({len(df):,} rows, {time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- prediction (used by game_server.py)

class DelayModel:
    def __init__(self, path=MODEL_FILE):
        saved = joblib.load(path)
        self.model, self.tables = saved["model"], saved["tables"]

    def predict(self, route, trip_id, stop_id, stop_position, sched_sec, date, now_sec,
                stop_seen=(), trip_seen=()):
        """now_sec: current time as seconds since the service day started (same clock as sched_sec).
        stop_seen: [(left_sec, delay)] departures of this route at this stop that already left today.
        trip_seen: [(left_sec, delay, stop_position)] stops this trip has already left."""
        row = pd.DataFrame({"date": [date], "route": [route], "trip_id": [trip_id], "stop_id": [stop_id],
                            "stop_position": [stop_position], "sched": [sched_sec]})
        row["dir"], row["trip_start"] = parse_trip(row["trip_id"])
        add_calendar(row)
        row["lead"] = float(np.clip(sched_sec - now_sec, 60, MAX_LEAD))

        stop_seen = sorted(s for s in stop_seen if s[0] <= now_sec)[-3:]
        trip_seen = sorted(t for t in trip_seen if t[0] <= now_sec and t[2] < stop_position)[-1:]
        nan = float("nan")
        row["stop_last"] = stop_seen[-1][1] if stop_seen else nan
        row["stop_last_age"] = now_sec - stop_seen[-1][0] if stop_seen else nan
        row["stop_last3"] = np.mean([s[1] for s in stop_seen]) if stop_seen else nan
        row["trip_last"] = trip_seen[0][1] if trip_seen else nan
        row["trip_last_age"] = now_sec - trip_seen[0][0] if trip_seen else nan
        row["trip_stops_since"] = stop_position - trip_seen[0][2] if trip_seen else nan

        row = apply_encodings(row, self.tables)
        return float(self.model.predict(row[FEATURES])[0])


if __name__ == "__main__":
    {"eval": evaluate, "train": train_and_save}[sys.argv[1] if len(sys.argv) > 1 else "eval"]()
