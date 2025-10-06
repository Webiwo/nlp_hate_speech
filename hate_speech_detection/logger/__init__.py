import datetime
import logging
import os

from matplotlib.dates import DAILY


LOG_FILE = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H')}-00-00.log"
DAILY_LOGS_DIR = f"{datetime.datetime.now().strftime('%Y-%m-%d')}"
logs_path = os.path.join(os.getcwd(), "logs", DAILY_LOGS_DIR)

os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

print(f"Log file path: {LOG_FILE_PATH}")

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s]: %(levelname)s: %(message)s",
    level=logging.DEBUG,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    filemode="a",
)
