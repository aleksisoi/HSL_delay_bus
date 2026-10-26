# gunicorn reads this file automatically, whatever start command the host uses.
# One worker only: live results and the guess resolver live in a single process.
# Threads let each open browser tab keep its live-updates connection.
import os

workers = 1
threads = 64
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
