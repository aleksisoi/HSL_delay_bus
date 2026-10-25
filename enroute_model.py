"""Add-on model for buses that are already running (from the bus_delay_models notebook).

The bus has just left stop A ("now") and we predict its delay at a stop 1-10 stops further on ("target").
It predicts the CHANGE in delay from A to the target and adds the current delay back. Inputs:
  * how the trip has gone so far: current delay, previous delays, recent averages and trend
  * the previous bus of the same route and direction at the target stop, and how its delay changed from A
  * historical average delays at stop A and at the target (computed from the other days when training)

Keys are route + direction + stop id, so everything can be rebuilt from live Digitransit data.

    python enroute_model.py eval    # train on all days but the last, report MAE on the last day
    python enroute_model.py train   # train on everything and save enroute_model.joblib
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
MODEL_FILE = "enroute_model.joblib"
MAX_AHEAD = 10             # stops ahead the model covers
MAX_AHEAD_AGE = 1800       # the previous bus counts only if it left the target at most 30 min ago
TRAIN_PAIRS = 4_000_000    # pairs used to fit the trees
RUSH_HOURS = {7, 8, 9, 15, 16, 17}

HIST = {"hour": ["route", "dir", "stop_id", "hour"], "stop": ["route", "dir", "stop_id"]}

FEATURES = ["pos_now", "sched_now", "hour", "weekday", "rush", "stops_in_trip",
            "delay_now", "prev_delay", "first_delay", "mean3", "mean5", "trend3",
            "pos_target", "stops_ahead", "sched_gap", "delay_trend", "hour_target",
            "ahead_delay", "ahead_age", "ahead_change",
            "hist_target_hour", "hist_now_hour", "hist_target_stop", "hist_now_stop",
            "hist_change_hour", "hist_change_stop"]


# ---------------------------------------------------------------- data

def load(db=DATA_DB):
    conn = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True)
    df = pd.read_sql_query(
        "SELECT service_date AS date, route, trip_id, stop_id, stop_position AS pos, "
        "scheduled_departure_sec AS sched, delay_seconds AS delay "
        "FROM stoptimes WHERE is_realtime = 1", conn)
    # Pattern length counts every stop of the trip, realtime or not (the live trip query knows them all).
    length = pd.read_sql_query("SELECT service_date AS date, trip_id, MAX(stop_position) + 1 AS stops_in_trip "
                               "FROM stoptimes GROUP BY 1, 2", conn)
    df = df.merge(length, on=["date", "trip_id"])
    df["dir"] = df["trip_id"].str.split("_").str[3]
    df["left"] = df["sched"] + df["delay"]
    df["hour"] = np.minimum(df["sched"] // 3600, 23)
    df["weekday"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.dayofweek
    return df.sort_values(["date", "trip_id", "pos"]).reset_index(drop=True)


def trip_history(df):
    """Per row: how the trip looked when the bus left this stop (only this and earlier realtime stops)."""
    g = df.groupby(["date", "trip_id"], sort=False)["delay"]
    n = df.groupby(["date", "trip_id"], sort=False).cumcount()
    cs = g.cumsum()
    out = pd.DataFrame(index=df.index)
    out["row"] = n
    out["prev_delay"] = g.shift(1)
    out["first_delay"] = g.transform("first")
    for k in (3, 5):
        before = cs.groupby([df["date"], df["trip_id"]], sort=False).shift(k).fillna(0)
        out[f"mean{k}"] = (cs - before) / np.minimum(n + 1, k)
    out["trend3"] = (df["delay"] - g.shift(3)) / 3
    return out


def hist_sums(df):
    """Per day sum and count of delays, so other-day averages are quick to compute."""
    return {name: df.groupby(["date"] + keys)["delay"].agg(["sum", "count"]) for name, keys in HIST.items()}


def hist_means(sums, exclude_day=None):
    out = {}
    for name, s in sums.items():
        if exclude_day is not None:
            s = s[s.index.get_level_values("date") != exclude_day]
        t = s.groupby(level=list(range(1, s.index.nlevels))).sum()
        out[name] = (t["sum"] / t["count"]).rename(f"hist_{name}")
    return out


def add_hist(pairs, means):
    for side, stop in (("target", "stop_t"), ("now", "stop_n")):
        hour = "hour_target" if side == "target" else "hour"
        h = means["hour"].rename(f"hist_{side}_hour")
        pairs = pairs.merge(h, left_on=["route", "dir", stop, hour], right_index=True, how="left")
        s = means["stop"].rename(f"hist_{side}_stop")
        pairs = pairs.merge(s, left_on=["route", "dir", stop], right_index=True, how="left")
    pairs["hist_change_hour"] = pairs["hist_target_hour"] - pairs["hist_now_hour"]
    pairs["hist_change_stop"] = pairs["hist_target_stop"] - pairs["hist_now_stop"]
    return pairs


def build_pairs(day, means):
    """All (now stop, target stop 1..10 ahead) pairs of one service day, with features and target."""
    day = day.reset_index(drop=True)
    day = pd.concat([day, trip_history(day)], axis=1)
    usable = day[day["row"] >= 1]             # like the notebook: a trip's first realtime stop is skipped

    now = usable.rename(columns={"pos": "pos_now", "sched": "sched_now", "delay": "delay_now",
                                 "left": "left_now", "stop_id": "stop_n"})
    now = now[["date", "route", "dir", "trip_id", "pos_now", "stop_n", "sched_now", "delay_now", "left_now",
               "hour", "weekday", "stops_in_trip", "prev_delay", "first_delay", "mean3", "mean5", "trend3"]]
    tgt = usable[["trip_id", "pos", "stop_id", "sched", "delay"]].rename(
        columns={"pos": "pos_target", "stop_id": "stop_t", "sched": "sched_target", "delay": "delay_target"})
    pairs = pd.concat([now.assign(pos_target=now["pos_now"] + k).merge(tgt, on=["trip_id", "pos_target"])
                       for k in range(1, MAX_AHEAD + 1)], ignore_index=True)

    pairs["rush"] = pairs["hour"].isin(RUSH_HOURS).astype(int)
    pairs["stops_ahead"] = pairs["pos_target"] - pairs["pos_now"]
    pairs["sched_gap"] = pairs["sched_target"] - pairs["sched_now"]
    pairs["delay_trend"] = pairs["delay_now"] - pairs["prev_delay"]
    pairs["hour_target"] = np.minimum(pairs["sched_target"] // 3600, 23)

    # The previous bus of the same route + direction that left the target stop before we left stop A.
    right = day[["route", "dir", "stop_id", "left", "delay", "trip_id"]].rename(
        columns={"stop_id": "stop_t", "left": "ahead_left", "delay": "ahead_delay", "trip_id": "ahead_trip"})
    pairs = pd.merge_asof(pairs.sort_values("left_now"), right.sort_values("ahead_left"),
                          left_on="left_now", right_on="ahead_left", by=["route", "dir", "stop_t"])
    pairs["ahead_age"] = pairs["left_now"] - pairs["ahead_left"]
    bad = (pairs["ahead_trip"] == pairs["trip_id"]) | (pairs["ahead_age"] > MAX_AHEAD_AGE)
    pairs.loc[bad, ["ahead_delay", "ahead_age", "ahead_trip"]] = np.nan
    at_a = day[["trip_id", "stop_id", "delay"]].rename(
        columns={"trip_id": "ahead_trip", "stop_id": "stop_n", "delay": "ahead_delay_at_now"})
    pairs = pairs.merge(at_a.drop_duplicates(["ahead_trip", "stop_n"]), on=["ahead_trip", "stop_n"], how="left")
    pairs["ahead_change"] = pairs["ahead_delay"] - pairs["ahead_delay_at_now"]

    return add_hist(pairs, means)


def all_pairs(df, days, sums, exclude_own_day, sample=None, seed=0):
    parts = []
    for d in days:
        p = build_pairs(df[df["date"] == d], hist_means(sums, d if exclude_own_day else None))
        if sample:
            p = p.sample(min(len(p), sample), random_state=seed)
        parts.append(p)
        print(f"  {d}: {len(p):,} pairs", end="\r")
    print()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- training

def fit(pairs):
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=600, learning_rate=0.1,
                                          max_leaf_nodes=127, min_samples_leaf=200,
                                          early_stopping=True, random_state=0)
    model.fit(pairs[FEATURES], pairs["delay_target"] - pairs["delay_now"])
    return model


def evaluate():
    t0 = time.time()
    df = load()
    days = sorted(df["date"].unique())
    train_days, test_day = days[:-1], days[-1]
    train_sums = hist_sums(df[df["date"] != test_day])
    per_day = TRAIN_PAIRS // len(train_days)
    train = all_pairs(df, train_days, train_sums, exclude_own_day=True, sample=per_day)
    model = fit(train)
    test = all_pairs(df, [test_day], train_sums, exclude_own_day=False)
    print(f"trained on {len(train):,} pairs from {train_days} in {time.time() - t0:.0f}s")

    y = test["delay_target"].to_numpy()
    pred = test["delay_now"].to_numpy() + model.predict(test[FEATURES])
    copy = test["delay_now"].to_numpy()
    print(f"\nMAE on {len(y):,} pairs, test day {test_day}:")
    print(f"  copy the current delay   {np.mean(np.abs(copy - y)):6.2f} s")
    print(f"  enroute model            {np.mean(np.abs(pred - y)):6.2f} s")
    print("  notebook on the same day: linear 36.02, network B 31.26")
    by_k = pd.DataFrame({"k": test["stops_ahead"], "copy": np.abs(copy - y), "enroute": np.abs(pred - y)})
    print("\nby stops ahead:\n" + by_k.groupby("k").mean().round(1).T.to_string())
    print(f"done in {time.time() - t0:.0f}s")


def train_and_save():
    t0 = time.time()
    df = load()
    days = sorted(df["date"].unique())
    sums = hist_sums(df)
    model = fit(all_pairs(df, days, sums, exclude_own_day=True, sample=TRAIN_PAIRS // len(days)))
    joblib.dump({"model": model, "hist": hist_means(sums)}, MODEL_FILE, compress=3)
    print(f"saved {MODEL_FILE} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- prediction (used by game_server.py)

class EnrouteModel:
    def __init__(self, path=MODEL_FILE):
        saved = joblib.load(path)
        self.model, self.hist = saved["model"], saved["hist"]

    def features(self, route, direction, date, stops_in_trip, passed, target, ahead=None):
        """passed: [{pos, stop_id, sched, delay, left}] realtime stops this trip has already left.
        target: {pos, stop_id, sched}.
        ahead: {left, delay, delay_at_now} for the previous bus of the route at the target stop, or None.
        Returns a one-row DataFrame, or None when the model doesn't apply."""
        passed = sorted(passed, key=lambda p: p["pos"])
        if len(passed) < 2:
            return None
        cur = passed[-1]
        if not 1 <= target["pos"] - cur["pos"] <= MAX_AHEAD:
            return None
        d = [p["delay"] for p in passed]
        hour = min(cur["sched"] // 3600, 23)
        nan = float("nan")
        row = {
            "route": route, "dir": direction, "stop_n": cur["stop_id"], "stop_t": target["stop_id"],
            "pos_now": cur["pos"], "sched_now": cur["sched"], "hour": hour,
            "weekday": pd.Timestamp(date).dayofweek, "rush": int(hour in RUSH_HOURS), "stops_in_trip": stops_in_trip,
            "delay_now": d[-1], "prev_delay": d[-2], "first_delay": d[0],
            "mean3": np.mean(d[-3:]), "mean5": np.mean(d[-5:]), "trend3": (d[-1] - d[-4]) / 3 if len(d) >= 4 else nan,
            "pos_target": target["pos"], "stops_ahead": target["pos"] - cur["pos"],
            "sched_gap": target["sched"] - cur["sched"], "delay_trend": d[-1] - d[-2],
            "hour_target": min(target["sched"] // 3600, 23),
            "ahead_delay": nan, "ahead_age": nan, "ahead_change": nan,
        }
        if ahead and 0 <= cur["left"] - ahead["left"] <= MAX_AHEAD_AGE:
            row["ahead_delay"], row["ahead_age"] = ahead["delay"], cur["left"] - ahead["left"]
            if ahead.get("delay_at_now") is not None:
                row["ahead_change"] = ahead["delay"] - ahead["delay_at_now"]
        return add_hist(pd.DataFrame([row]), self.hist)

    def predict(self, *args, **kwargs):
        """Predicted delay (seconds) at the target stop, or None when the model doesn't apply."""
        row = self.features(*args, **kwargs)
        if row is None:
            return None
        return float(row["delay_now"].iloc[0] + self.model.predict(row[FEATURES])[0])


if __name__ == "__main__":
    {"eval": evaluate, "train": train_and_save}[sys.argv[1] if len(sys.argv) > 1 else "eval"]()
