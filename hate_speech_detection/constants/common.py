import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT_DIR = os.getcwd()
CONFIG_FILE_PATH = os.path.join(
    ROOT_DIR, "hate_speech_detection", "configuration", "config.yaml"
)
MAIN_ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
# MAIN_ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts", TIMESTAMP)
