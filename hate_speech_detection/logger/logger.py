import os
import shutil
import datetime
import logging
from logging.handlers import TimedRotatingFileHandler


def clean_old_logs(base_log_dir: str, days_to_keep: int = 7):
    """
    Deletes log directories older than the specified number of days.
    """

    if not os.path.exists(base_log_dir):
        return
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)

    for folder in os.listdir(base_log_dir):
        folder_path = os.path.join(base_log_dir, folder)
        if os.path.isdir(folder_path):
            try:
                folder_date = datetime.datetime.strptime(folder, "%Y-%m-%d")
                if folder_date < cutoff_date:
                    shutil.rmtree(folder_path)
                    print(f"Removed old log folder: {folder_path}")
            except ValueError:
                # Folder does not match the date format, ignore
                continue


def get_logger(name: str = "hate_speech_detection") -> logging.Logger:
    """
    Creates a logger with hourly rotation and daily log directories.
    """

    base_log_dir = os.path.join(os.getcwd(), "logs")
    clean_old_logs(base_log_dir, days_to_keep=7)
    current_day_dir = os.path.join(
        base_log_dir, datetime.datetime.now().strftime("%Y-%m-%d")
    )
    os.makedirs(current_day_dir, exist_ok=True)

    # Path of the main log file (e.g. logs/2025-10-10/app.log)
    log_file_path = os.path.join(current_day_dir, "app.log")

    # Log file rotation configuration
    file_handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when="H",
        interval=1,
        backupCount=48,
        encoding="utf-8",
        utc=False,
    )

    # Log message format
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # Console handler for output to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = get_logger()
logger.info("Logger running. Older logs are cleared automatically.")
