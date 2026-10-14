import os
import sqlite3
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import requests

SUBSCRIPTION_KEY =os.environ.get("DIGITRANSIT_KEY")
URL = "https://api.digitransit.fi/routing/v2/hsl/gtfs/v1"

TRACK_ALL_BUSES = True       # True = all routes
ROUTE_IDS = []              # only used when TRACK_ALL_BUSES = False

MAX_ROUTES = None            # set e.g. 10 to test with a few routes before the full run

OUTPUT_CSV = "route_all.csv"   
DB_FILE = "route_all.db"       
DURATION_SECONDS = 60 * 15  # total run time 
POLL_INTERVAL_SECONDS = 60 * 5  # minimum time between full sweeps of all routes
REQUEST_DELAY_SECONDS = 0.5     # pause between API calls to stay under the rate limit

ONLY_DEPARTED = False #Set True to only get realtime data

TZ = ZoneInfo("Europe/Helsinki")

ROUTES_QUERY = """
{
  routes(feeds: ["HSL"], transportModes: [BUS]) {
    gtfsId
    shortName
  }
}
"""

ROUTE_QUERY_TEMPLATE = """
{
  route(id: "%s") {
    shortName
    mode
    patterns {
      code
      tripsForDate(serviceDate: "%s") {
        gtfsId
        stoptimesForDate(serviceDate: "%s") {
          scheduledDeparture
          realtimeDeparture
          departureDelay
          realtime
          stopPositionInPattern
          stop {
            gtfsId
            name
          }
        }
      }
    }
  }
}
"""


def post_query(session, query):
    """POST a GraphQL query, retrying on rate limits and server errors."""
    for attempt in range(4):
        try:
            response = session.post(URL, json={"query": query}, timeout=30)
            if response.status_code in (429, 502, 503, 504):
                wait = 2 ** attempt
                print(f"  HTTP {response.status_code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            return response.json()
        except Exception as e:
            print(f"  Request failed ({e}), retrying")
            time.sleep(2 ** attempt)
    return None


def discover_bus_routes(session):
    data = post_query(session, ROUTES_QUERY)
    if not data or "errors" in data:
        print(f"Could not list bus routes: {data.get('errors') if data else 'no response'}")
        return []
    routes = data.get("data", {}).get("routes") or []
    ids = sorted({r["gtfsId"] for r in routes})
    print(f"Found {len(ids)} bus routes")
    return ids


def fetch_route_journeys(session, route_id, service_date, seconds_now):
    query = ROUTE_QUERY_TEMPLATE % (route_id, service_date, service_date)
    data = post_query(session, query)
    if data is None:
        return []
    if "errors" in data:
        print(f"API error for {route_id}: {data['errors']}")
        return []

    route = (data.get("data") or {}).get("route")
    if route is None:
        print(f"No route found for id {route_id} (check the ID is correct)")
        return []

    route_name = route["shortName"]
    rows = []

    for pattern in route["patterns"]:
        for trip in pattern["tripsForDate"]:
            trip_id = trip["gtfsId"]
            for st in trip["stoptimesForDate"]:
                scheduled_dep = st["scheduledDeparture"]
                realtime_dep = st["realtimeDeparture"]
                delay = st["departureDelay"]  

                if ONLY_DEPARTED:
                    effective = realtime_dep if realtime_dep is not None else scheduled_dep
                    if effective > seconds_now:
                        continue

                rows.append({
                    "service_date": service_date,  
                    "route": route_name,
                    "trip_id": trip_id,
                    "stop_position": st["stopPositionInPattern"],
                    "stop_id": st["stop"]["gtfsId"],
                    "scheduled_departure_sec": scheduled_dep,
                    "delay_seconds": delay,
                    "is_realtime": st["realtime"],
                })
    return rows


def open_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stoptimes (
            service_date TEXT,
            route TEXT,
            trip_id TEXT,
            stop_position INTEGER,
            stop_id TEXT,
            scheduled_departure_sec INTEGER,
            delay_seconds INTEGER,
            is_realtime INTEGER,
            PRIMARY KEY (service_date, trip_id, stop_position)
        )
    """)
    return conn


UPSERT_SQL = """
INSERT INTO stoptimes (
    service_date, route, trip_id, stop_position, stop_id,
    scheduled_departure_sec, delay_seconds, is_realtime
) VALUES (
    :service_date, :route, :trip_id, :stop_position, :stop_id,
    :scheduled_departure_sec, :delay_seconds, :is_realtime
)
ON CONFLICT (service_date, trip_id, stop_position) DO UPDATE SET
    delay_seconds = excluded.delay_seconds,
    is_realtime = excluded.is_realtime
WHERE
    -- never replace real delay data with a schedule-only row
    (excluded.is_realtime = 1 OR stoptimes.is_realtime = 0)
    -- and only touch the row if something actually changed
    AND (
        stoptimes.delay_seconds IS NOT excluded.delay_seconds
        OR stoptimes.is_realtime IS NOT excluded.is_realtime
    )
"""


def upsert_rows(conn, rows):
    """Insert new stop-times, update changed ones, skip identical ones. Returns rows touched."""
    if not rows:
        return 0
    for r in rows:
        r["is_realtime"] = 1 if r["is_realtime"] else 0
    before = conn.total_changes
    conn.executemany(UPSERT_SQL, rows)
    conn.commit()
    return conn.total_changes - before


def export_csv(conn):
    df = pd.read_sql_query(
        "SELECT route, trip_id, stop_position, stop_id, scheduled_departure_sec, "
        "delay_seconds, is_realtime FROM stoptimes "
        "ORDER BY service_date, route, trip_id, stop_position", conn)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Exported {len(df)} rows to {OUTPUT_CSV}")

def main():
    if not SUBSCRIPTION_KEY:
        raise SystemExit("Set the DIGITRANSIT_KEY environment variable first.")

    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "digitransit-subscription-key": SUBSCRIPTION_KEY,
    })
    
    route_ids = discover_bus_routes(session) if TRACK_ALL_BUSES else list(ROUTE_IDS)
    if MAX_ROUTES:
        route_ids = route_ids[:MAX_ROUTES]
    if not route_ids:
        raise SystemExit("No routes to track.")

    conn = open_db()
    start_time = time.time()
    poll_count = 0
   
    print(F"Polling date: {datetime.now(TZ)}")
    print(f"Tracking {len(route_ids)} route(s) for {DURATION_SECONDS}s, "
          f"sweeping at most every {POLL_INTERVAL_SECONDS}s")

    if datetime.now(TZ).hour < 17: #This will poll for yesterdays night busses, last shift around 5am and records are kept for 12 hours, so we need to poll for yesterdays data if its before 5pm
        print("\nPolling for yesterdays data\n")
        while time.time() - start_time < DURATION_SECONDS:
            poll_count += 1
            sweep_start = time.time()
            now = datetime.now(TZ)
            yesterday = datetime.now(TZ) - pd.Timedelta(days=1)
            service_date = yesterday.strftime("%Y%m%d")  # yesterday, as YYYYMMDD
            seconds_now = now.hour * 3600 + now.minute * 60 + now.second


            logged = 0
            for i, route_id in enumerate(route_ids, 1):
                rows = fetch_route_journeys(session, route_id, service_date, seconds_now)

                logged += upsert_rows(conn, rows)

                if i % 25 == 0:
                    print(f"  sweep {poll_count}: {i}/{len(route_ids)} routes, {logged} new/changed rows")
                time.sleep(REQUEST_DELAY_SECONDS)

            print(f"Sweep {poll_count} done in {time.time() - sweep_start:.0f}s: "
                f"{logged} new or changed stop-times")

            remaining = POLL_INTERVAL_SECONDS - (time.time() - sweep_start)
            if remaining > 0 and time.time() - start_time + remaining < DURATION_SECONDS:
                time.sleep(remaining)
        start_time = time.time()
    

    print("\nPolling for todays data\n")
    while time.time() - start_time < DURATION_SECONDS:
        poll_count += 1
        sweep_start = time.time()
        now = datetime.now(TZ)
        service_date = now.strftime("%Y%m%d")  # today, as YYYYMMDD
        seconds_now = now.hour * 3600 + now.minute * 60 + now.second


        logged = 0
        for i, route_id in enumerate(route_ids, 1):
            rows = fetch_route_journeys(session, route_id, service_date, seconds_now)

            logged += upsert_rows(conn, rows)

            if i % 25 == 0:
                print(f"  sweep {poll_count}: {i}/{len(route_ids)} routes, {logged} new/changed rows")
            time.sleep(REQUEST_DELAY_SECONDS)

        print(f"Sweep {poll_count} done in {time.time() - sweep_start:.0f}s: "
              f"{logged} new or changed stop-times")

        remaining = POLL_INTERVAL_SECONDS - (time.time() - sweep_start)
        if remaining > 0 and time.time() - start_time + remaining < DURATION_SECONDS:
            time.sleep(remaining)
    

    print(f"\nDone. Finished after {poll_count} sweeps.")
    export_csv(conn)
    conn.close()


if __name__ == "__main__":
    main()