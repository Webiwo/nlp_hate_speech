import subprocess
import os
import sys
from hate_speech_detection.exception.exception import GCloudSyncError
from hate_speech_detection.logger import logging


class GCloudSync:
    def sync_folder_to_gcloud(self, gcp_bucket_url, folder_path):
        """
        Syncs a local folder to a Google Cloud Storage bucket using the gsutil command.

        Args:
            gcp_bucket_url (str): The URL of the GCP bucket (e.g., gs://your-bucket-name).
            folder_path (str): The path to the local folder to be synced.

        Returns:
            None
        """
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(
                    f"The folder path {folder_path} does not exist or is not a directory."
                )
            # Construct the gsutil command
            command = [
                "gcloud.cmd",
                "storage",
                "rsync",
                "--recursive",
                folder_path,
                f"gs://{gcp_bucket_url}",
            ]

            # Execute the command
            subprocess.run(command, check=True)
            logging.info(f"Successfully synced {folder_path} to {gcp_bucket_url}")

        except Exception as e:
            raise GCloudSyncError(e) from e

    def sync_folder_from_gcloud(self, gcp_bucket_url, folder_path):
        """
        Syncs a Google Cloud Storage bucket to a local folder using the gsutil command.

        Args:
            gcp_bucket_url (str): The URL of the GCP bucket (e.g., gs://your-bucket-name).
            folder_path (str): The path to the local folder to be synced.

        Returns:
            None
        """

        os.makedirs(folder_path, exist_ok=True)

        try:
            # Construct the gsutil command
            command = [
                "gcloud.cmd",
                "storage",
                "rsync",
                "--recursive",
                f"gs://{gcp_bucket_url}",
                folder_path,
            ]

            # Execute the command
            subprocess.run(command, check=True)
            logging.info(f"Successfully synced {gcp_bucket_url} to {folder_path}")

        except Exception as e:
            raise GCloudSyncError(e) from e
