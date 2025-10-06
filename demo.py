from hate_speech_detection.exception import CustomException
from hate_speech_detection.configuration.gcloud_syncer import GCloudSync


# def divide(a, b):
#     return a / b


# def run_divide():
#     return divide(3, 0)


# try:
#     run_divide()
# except Exception as e:
#     ce = CustomException(e)
#     print(ce)


sync = GCloudSync()
sync.sync_folder_from_gcloud("hate-speech-detection", ".")
