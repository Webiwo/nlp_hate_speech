from zipfile import ZipFile, BadZipFile
from hate_speech_detection.exception.exception import (
    DataIngestionError,
    GCloudSyncError,
)
from hate_speech_detection.logger.logger import logger
from hate_speech_detection.configuration.gcloud_syncer import GCloudSync
from hate_speech_detection.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.config = data_ingestion_config
        self.gcloud = GCloudSync()

    def get_data_from_gcloud(self):
        try:
            logger.info(
                f"Downloading data from gcloud bucket: {self.config.bucket_name} to {self.config.artifacts_dir} directory"
            )
            self.gcloud.sync_folder_from_gcloud(
                self.config.bucket_name,
                self.config.artifacts_dir,
            )
        except Exception as e:
            raise GCloudSyncError(e) from e

    def unzip_dataset(self):
        logger.info(
            f"Unzipping file {self.config.zip_file_name} to {self.config.artifacts_dir} directory"
        )
        try:
            with ZipFile(self.config.zip_file_name, "r") as zip_ref:
                zip_ref.extractall(self.config.artifacts_dir)
        except (FileNotFoundError, BadZipFile) as e:
            raise DataIngestionError(e) from e

    def initiate_data_ingestion(self):
        try:
            self.get_data_from_gcloud()
            self.unzip_dataset()
            logger.info("Data ingestion completed successfully.")
        except Exception as e:
            raise DataIngestionError(e) from e
