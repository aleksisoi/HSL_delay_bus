https://hsl-delay-game.onrender.com/ to launch the browser game. Due to page limitations first visitors every 15 minutes need to wait for the page to be redeployed and models loaded. 
To run locally download game_server.py, static folder with index inside, 2 joblib files which are the models, start with `python game_server.py` (needs `DIGITRANSIT_KEY` in `.env` at the current moment). 
HSLroutesFINAL.py can be used to query routes, needs DIGITRANSIT_KEY.
query_route can be used to slice and view data.
Latest dataset is in releases. 

INSTRUCTIONS FOR THE GAME:

Load the page and type in the route you wish to place your delay guess on. The route numbers are in the same form as they appear in HSL-app. 
Choose a stop on the highlighted route and the departure time.
Place your guess of delay in SECONDS. Positive for being late and negative for being ahead schedule.
You will see how other models have estimated the delay and receive results when the bus passes said stop.

Models we have trained:
enroute_model.py, the bus is already running, 1-10 stops before the guessed stop
delay_model.py, everything else, e.g. guesses before the trip has started
lightgbm_model.py, additional model which we were unable to add to the game but stands in as a legacy option

Page hosted by Render
