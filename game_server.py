import json, os, queue, re, sqlite3, threading, time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory

import delay_model
import enroute_model

load_dotenv()
KEY = os.environ.get("DIGITRANSIT_KEY")
if not KEY:
    raise SystemExit("Set DIGITRANSIT_KEY (env var or .env file).")
URL = "https://api.digitransit.fi/routing/v2/hsl/gtfs/v1"
TZ = ZoneInfo("Europe/Helsinki")
DATA_DB = os.environ.get("DELAY_DB", "route_all.db")   
GAME_DB = os.environ.get("GAME_DB", "game.db")          
app = Flask(__name__, static_folder="static")
CACHE, MODEL = {}, {}
subs, lock = [], threading.Lock()


def gql(q, v=None):
    r = requests.post(URL, json={"query": q, "variables": v or {}}, timeout=30,
                      headers={"digitransit-subscription-key": KEY})
    r.raise_for_status()
    d = r.json()
    if d.get("errors"):
        raise RuntimeError(d["errors"])
    return d["data"]


def load_model():
    """Load the saved models, training them from the logger database first if they don't exist yet.
    delay_model is required; enroute_model (buses already running) is optional."""
    for module, required in ((delay_model, True), (enroute_model, False)):
        if not os.path.exists(module.MODEL_FILE):
            if not os.path.exists(DATA_DB):
                if required:
                    raise SystemExit(f"{os.path.abspath(DATA_DB)} not found. Set DELAY_DB to the full path of your route_all.db.")
                print(f"{module.MODEL_FILE} missing: predictions for running buses use the general model")
                continue
            print(f"no {module.MODEL_FILE}, training it (takes several minutes)...")
            module.train_and_save()
    MODEL["m"] = delay_model.DelayModel()
    if os.path.exists(enroute_model.MODEL_FILE):
        MODEL["en"] = enroute_model.EnrouteModel()
    print("models loaded:", ", ".join(MODEL))


TRIP_Q = """query($t:String!,$d:String!){trip(id:$t){stoptimesForDate(serviceDate:$d){
  scheduledDeparture realtimeDeparture realtime stopPositionInPattern stop{gtfsId}}}}"""


def trip_stoptimes(trip_id, date):
    """Every stop of the trip: position, stop id, schedule and (realtime) departure, as seconds of the service day."""
    trip = gql(TRIP_Q, {"t": trip_id, "d": date})["trip"] or {}
    return [{"pos": t["stopPositionInPattern"], "stop_id": t["stop"]["gtfsId"], "sched": t["scheduledDeparture"],
             "left": t["realtimeDeparture"], "delay": t["realtimeDeparture"] - t["scheduledDeparture"],
             "realtime": t["realtime"]}
            for t in trip.get("stoptimesForDate") or []]


def direction(trip_id):
    parts = trip_id.split("_")
    return parts[3] if len(parts) > 3 else None


def predict_enroute(g, stops, trip, service_day, now_sec):
    """Prediction of the add-on model when the bus is already running 1-10 stops before the guessed stop, else None."""
    if "en" not in MODEL:
        return None
    passed = [t for t in trip if t["realtime"] and t["left"] <= now_sec and t["pos"] < g["position"]]
    if len(passed) < 2:
        return None
    cur = max(passed, key=lambda t: t["pos"])
    if not 1 <= g["position"] - cur["pos"] <= enroute_model.MAX_AHEAD:
        return None

    # The previous bus of the same route and direction that left the guessed stop before this one left stop A.
    d = direction(g["trip_id"])
    earlier = [s for s in stops if s["route_id"] == g["route_id"] and s["delay"] is not None
               and s["trip"] != g["trip_id"] and direction(s["trip"]) == d
               and s["left_epoch"] - service_day <= cur["left"]]
    ahead = None
    if earlier:
        a = max(earlier, key=lambda s: s["left_epoch"])
        at_a = next((s for s in stop_times(cur["stop_id"], g["service_date"]) if s["trip"] == a["trip"]), None)
        ahead = {"left": a["left_epoch"] - service_day, "delay": a["delay"],
                 "delay_at_now": at_a["delay"] if at_a else None}

    return MODEL["en"].predict(g["route"], d, g["service_date"], len(trip), passed,
                               {"pos": g["position"], "stop_id": g["stop_id"], "sched": g["sched_sec"]}, ahead)


def predict(g):
    service_day = g["sched_epoch"] - g["sched_sec"]
    now_sec = time.time() - service_day
    try:  # live data: what this stop and this trip have looked like today
        stops = stop_times(g["stop_id"], g["service_date"])
        trip = trip_stoptimes(g["trip_id"], g["service_date"])
    except Exception as e:
        print("live data unavailable:", e)
        stops, trip = [], []
    try:
        en = predict_enroute(g, stops, trip, service_day, now_sec)
        if en is not None:
            return en
    except Exception as e:
        print("enroute model failed, using the general model:", e)
    stop_seen = [(s["left_epoch"] - service_day, s["delay"]) for s in stops
                 if s["route_id"] == g["route_id"] and s["delay"] is not None]
    trip_seen = [(t["left"], t["delay"], t["pos"]) for t in trip if t["realtime"]]
    return MODEL["m"].predict(g["route"], g["trip_id"], g["stop_id"], g["position"], g["sched_sec"],
                              g["service_date"], now_sec, stop_seen, trip_seen)


# ---------- game storage + broadcast ----------
def gdb():
    c = sqlite3.connect(GAME_DB, timeout=10)
    c.row_factory = sqlite3.Row
    return c


with gdb() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS guesses (id INTEGER PRIMARY KEY, player TEXT, route TEXT,
        route_id TEXT, stop_id TEXT, stop_name TEXT, trip_id TEXT, service_date TEXT, sched_epoch INTEGER,
        human REAL, model REAL, actual REAL, status TEXT, created INTEGER)""")


def broadcast(ev):
    with lock:
        for q in subs:
            q.put(ev)


@app.get("/api/events")
def events():
    q = queue.Queue()
    with lock:
        subs.append(q)

    def gen():
        try:
            while True:
                try:
                    yield f"data: {json.dumps(q.get(timeout=20))}\n\n"
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            with lock:
                subs.remove(q)
    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


STOP_Q = """query($s:String!,$d:String!){stop(id:$s){stoptimesForServiceDate(date:$d,omitNonPickups:true){
  pattern{headsign route{gtfsId}} stoptimes{serviceDay scheduledDeparture realtimeDeparture departureDelay
  realtime stopPositionInPattern trip{gtfsId}}}}}"""


def stop_times(stop_id, date):
    out = []
    for p in gql(STOP_Q, {"s": stop_id, "d": date})["stop"]["stoptimesForServiceDate"]:
        for t in p["stoptimes"]:
            out.append({"route_id": p["pattern"]["route"]["gtfsId"], "headsign": p["pattern"]["headsign"],
                        "trip": t["trip"]["gtfsId"], "epoch": t["serviceDay"] + t["scheduledDeparture"],
                        "sched_sec": t["scheduledDeparture"], "position": t["stopPositionInPattern"],
                        "left_epoch": t["serviceDay"] + (t["realtimeDeparture"] or t["scheduledDeparture"]),
                        "delay": t["departureDelay"] if t["realtime"] else None})
    return sorted(out, key=lambda x: x["epoch"])

@app.get("/api/routes")
def routes():
    if "routes" not in CACHE:
        d = gql('{routes(feeds:["HSL"],transportModes:[BUS]){gtfsId shortName longName}}')
        CACHE["routes"] = sorted(d["routes"], key=lambda r: r["shortName"])
    return jsonify(CACHE["routes"])


@app.get("/api/route/<rid>")
def route(rid):
    return jsonify(gql("""query($i:String!){route(id:$i){shortName patterns{code headsign
        patternGeometry{points} stops{gtfsId name lat lon}}}}""", {"i": rid})["route"])


@app.get("/api/timetable")
def timetable():
    date = datetime.now(TZ).strftime("%Y%m%d")
    rows = [t for t in stop_times(request.args["stop"], date) if t["route_id"] == request.args["route"]]
    for t in rows:
        t["past"] = t["left_epoch"] <= time.time()
    return jsonify({"date": date, "now": time.time(), "times": rows})


@app.post("/api/guess")
def guess():
    g = request.get_json()
    mdl = predict(g)
    with gdb() as c:
        cur = c.execute("""INSERT INTO guesses (player, route, route_id, stop_id, stop_name, trip_id, service_date,
            sched_epoch, human, model, status, created) VALUES (?,?,?,?,?,?,?,?,?,?, 'open', ?)""",
                        (g["player"][:24] or "anon", g["route"], g["route_id"], g["stop_id"], g["stop_name"],
                         g["trip_id"], g["service_date"], g["sched_epoch"], g["human"], mdl, int(time.time())))
    broadcast({"type": "guess", "id": cur.lastrowid})
    return jsonify({"id": cur.lastrowid, "model": mdl})


@app.get("/api/guesses")
def guesses():
    with gdb() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM guesses ORDER BY id DESC LIMIT 40")]
        done = [dict(r) for r in c.execute("SELECT human, model, actual FROM guesses WHERE status='done'")]
    n = len(done)
    mae = lambda k: round(sum(abs(d[k] - d["actual"]) for d in done) / n, 1) if n else None
    wins = lambda a, b: sum(abs(d[a] - d["actual"]) < abs(d[b] - d["actual"]) for d in done)
    return jsonify({"rows": rows, "stats": {"resolved": n, "human_mae": mae("human"), "model_mae": mae("model"),
                                            "human_wins": wins("human", "model"), "model_wins": wins("model", "human")}})


def resolve(g):
    t = next((x for x in stop_times(g["stop_id"], g["service_date"]) if x["trip"] == g["trip_id"]), None)
    if t and t["delay"] is not None and t["left_epoch"] <= time.time():
        eh, em = abs(g["human"] - t["delay"]), abs(g["model"] - t["delay"])
        status, actual = "done", t["delay"]
    elif time.time() > g["sched_epoch"] + 3 * 3600:
        status, actual = "expired", None
    else:
        return
    with gdb() as c:
        c.execute("UPDATE guesses SET status=?, actual=? WHERE id=?", (status, actual, g["id"]))
    if status == "done":
        broadcast({"type": "result", "id": g["id"], "player": g["player"], "route": g["route"],
                   "stop": g["stop_name"], "time": g["sched_epoch"], "human": g["human"], "model": g["model"],
                   "actual": actual, "winner": "human" if eh < em else "model" if em < eh else "tie"})
    else:
        broadcast({"type": "guess", "id": g["id"]})


def resolver():
    while True:
        time.sleep(30)
        try:
            with gdb() as c:
                open_ = [dict(r) for r in c.execute(
                    "SELECT * FROM guesses WHERE status='open' AND sched_epoch < ?", (time.time() + 120,))]
            for g in open_:
                resolve(g)
        except Exception as e:
            print("resolver:", e)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")



load_model()
threading.Thread(target=resolver, daemon=True).start()

if __name__ == "__main__":
    app.run(port=5000, threaded=True)