dataset comes in 2 formats; .db and .csv. Files to be downloaded at: 
use routefinder to query single route, in future possibly other parameters. Preferred file for operations .db

TO BE DONE:  active routes per hour, heatmap of average delay per stop per hour, add traffic, weekend, night parameters, make jupyter-file for routefinder
            

## Guess the HSL delay (web game)

`game_server.py` serves the game in `static/`. Players guess a bus's delay and compete against two models:

| Model | Used when | Test MAE (Oct 17) |
|---|---|---|
| `enroute_model.py` | the bus is already running, 1-10 stops before the guessed stop | 32.0 s (copying the current delay: 44.9 s) |
| `delay_model.py` | everything else, e.g. guesses before the trip has started | 64.2 s (old mixed model: 78.3 s) |

Retrain both after logging more data (needs `route_all.db`), then commit the two `.joblib` files:

```bash
python delay_model.py eval      # optional: score on the last logged day
python delay_model.py train     # -> delay_model.joblib   (~2 min)
python enroute_model.py eval    # optional
python enroute_model.py train   # -> enroute_model.joblib (~5 min)
```

Run locally with `python game_server.py` (needs `DIGITRANSIT_KEY` in `.env`). Render setup is in `render.yaml`.
