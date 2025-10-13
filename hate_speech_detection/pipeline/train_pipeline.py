from hate_speech_detection.components.data_transforamation import DataTransformation
from hate_speech_detection.configuration.config_manager import ConfigurationManager
from hate_speech_detection.logger.logger import logger
from hate_speech_detection.exception.exception import PipelineExecutionError

from hate_speech_detection.components.data_ingestion import DataIngestion
from hate_speech_detection.components.data_validator import DataValidator


class TrainPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.ingest_config = self.config_manager.get_data_ingestion_config()
        self.trans_config = self.config_manager.get_data_transformation_config()

    def _run_data_ingestion(self):
        try:
            # Data Ingestion
            data_in = DataIngestion(self.ingest_config)
            data_in.initiate_data_ingestion()

            # Data validation
            validator = DataValidator(file_path=self.ingest_config.imbalanced_data_path)
            validator.validate()
            validator = DataValidator(file_path=self.ingest_config.raw_data_path)
            validator.validate()

            # Data cleaning and transformation
            transformator = DataTransformation(self.trans_config, self.ingest_config)
            transformator.initiate_data_transformation()

        except Exception as e:
            logger.error(f"Unexpected pipeline error: {e}")
            raise PipelineExecutionError(e) from e

    def run_pipeline(self):
        try:
            logger.info("Starting training pipeline...")
            self._run_data_ingestion()
            logger.info("Training pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}")
            raise PipelineExecutionError(e) from e
