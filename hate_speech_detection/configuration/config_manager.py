from calendar import c
import os
from re import T
import yaml
from dataclasses import dataclass
from hate_speech_detection.constants import CONFIG_FILE_PATH, MAIN_ARTIFACTS_DIR
from hate_speech_detection.utils.common_utils import read_yaml, create_directories
from hate_speech_detection.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from hate_speech_detection.exception.exception import (
    PipelineExecutionError,
    DataValidationError,
)
from hate_speech_detection.logger.logger import logger


@dataclass
class Configuration:
    data_ingestion: dict
    data_transformation: dict
    model_trainer: dict


class ConfigurationManager:
    """Manages reading and validation of YAML configurations"""

    def __init__(self, config_file_path: str = None):
        try:
            self.config_file_path = config_file_path or CONFIG_FILE_PATH
            self.main_artifacts_dir = MAIN_ARTIFACTS_DIR
            self.data_ingestion_dir = ""
            self.data_transformation_dir = ""
            self.model_trainer_dir = ""
            self.config = None
            self._load_config()
            self._create_directories()
        except Exception as e:
            logger.exception("Error initializing ConfigurationManager")
            raise PipelineExecutionError(e) from e

    def _load_config(self):
        """Load and validate YAML configuration file"""
        try:
            config_dict = read_yaml(self.config_file_path)
            if not isinstance(config_dict, dict):
                raise DataValidationError("YAML is empty or not properly formatted.")

            self.config = Configuration(
                data_ingestion=config_dict.get("data_ingestion", {}),
                data_transformation=config_dict.get("data_transformation", {}),
                model_trainer=config_dict.get("model_trainer", {}),
            )

            logger.info(
                f"Configuration from file: {self.config_file_path} has been loaded successfully."
            )

        except (FileNotFoundError, yaml.YAMLError, DataValidationError) as e:
            logger.exception("Error loading YAML configuration")
            raise PipelineExecutionError(e)

    def _create_directories(self):
        """
        Creates all artifact directories needed by the pipeline.
        """
        self.data_ingestion_dir = os.path.join(
            self.main_artifacts_dir, self.config.data_ingestion["artifacts_dir"]
        )
        self.data_transformation_dir = os.path.join(
            self.main_artifacts_dir, self.config.data_transformation["artifacts_dir"]
        )
        self.model_trainer_dir = os.path.join(
            self.main_artifacts_dir, self.config.model_trainer["artifacts_dir"]
        )

        try:
            dirs_to_create = [
                self.main_artifacts_dir,
                self.data_ingestion_dir,
                self.data_transformation_dir,
                self.model_trainer_dir,
            ]

            create_directories(dirs_to_create)

        except Exception as e:
            logger.exception("Error creating artifact directories")
            raise PipelineExecutionError(e)

    #
    # Methods to get specific component configurations
    #

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            bucket_name=self.config.data_ingestion["bucket_name"],
            artifacts_dir=self.data_ingestion_dir,
            zip_file_name=os.path.join(
                self.data_ingestion_dir, self.config.data_ingestion["zip_file_name"]
            ),
            imbalanced_data_path=os.path.join(
                self.data_ingestion_dir, self.config.data_ingestion["imbalanced_file"]
            ),
            raw_data_path=os.path.join(
                self.data_ingestion_dir, self.config.data_ingestion["raw_file"]
            ),
        )

    def get_data_transformation_config(self):
        return DataTransformationConfig(
            artifacts_dir=self.data_transformation_dir,
            transformed_file_path=os.path.join(
                self.data_transformation_dir,
                self.config.data_transformation["transformed_file_name"],
            ),
            drop_columns=self.config.data_transformation["drop_columns"],
            class_column=self.config.data_transformation["class_column"],
            label_column=self.config.data_transformation["label_column"],
            tweet_column=self.config.data_transformation["tweet_column"],
            language=self.config.data_transformation["language"],
            more_stopwords=self.config.data_transformation["more_stopwords"],
        )

    def get_model_trainer_config(self):
        return ModelTrainerConfig(
            artifacts_dir=self.model_trainer_dir,
            tokenizer_path=os.path.join(
                self.model_trainer_dir, self.config.model_trainer["tokenizer_name"]
            ),
            trained_model_path=os.path.join(
                self.model_trainer_dir, self.config.model_trainer["trained_model_name"]
            ),
            x_train_path=os.path.join(
                self.model_trainer_dir, self.config.model_trainer["x_train_file"]
            ),
            y_train_path=os.path.join(
                self.model_trainer_dir, self.config.model_trainer["y_train_file"]
            ),
            x_test_path=os.path.join(
                self.model_trainer_dir, self.config.model_trainer["x_test_file"]
            ),
            y_test_path=os.path.join(
                self.model_trainer_dir, self.config.model_trainer["y_test_file"]
            ),
            random_state=self.config.model_trainer["random_state"],
            epochs=self.config.model_trainer["epochs"],
            batch_size=self.config.model_trainer["batch_size"],
            test_split=self.config.model_trainer["test_split"],
            max_words=self.config.model_trainer["max_words"],
            max_len=self.config.model_trainer["max_len"],
            loss=self.config.model_trainer["loss"],
            metrics=self.config.model_trainer["metrics"],
            activation=self.config.model_trainer["activation"],
        )
