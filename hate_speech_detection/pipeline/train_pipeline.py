from hate_speech_detection.configuration.config_manager import ConfigurationManager
from hate_speech_detection.exception.exception import PipelineExecutionError
from hate_speech_detection.logger.logger import logger

from hate_speech_detection.components.data_transforamation import DataTransformation
from hate_speech_detection.components.model_evaluation import ModelEvaluation
from hate_speech_detection.components.data_ingestion import DataIngestion
from hate_speech_detection.components.data_validator import DataValidator
from hate_speech_detection.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self, configuration_manager=None):
        self.config_manager = configuration_manager or ConfigurationManager()
        self.ingest_config = self.config_manager.get_data_ingestion_config()
        self.trans_config = self.config_manager.get_data_transformation_config()
        self.train_config = self.config_manager.get_model_trainer_config()
        self.eval_config = self.config_manager.get_model_evaluation_config()
        self.pred_config = self.config_manager.get_prediction_config()

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

    def _train_model(self):
        try:
            trainer = ModelTrainer(self.train_config, self.trans_config)
            trainer.initiate_model_trainer()
        except Exception as e:
            logger.error(f"Unexpected training error: {e}")
            raise PipelineExecutionError(e) from e

    def _evaluate_model(self):
        try:
            eval = ModelEvaluation(
                self.eval_config, self.train_config, self.trans_config, self.pred_config
            )
            eval.initiate_model_evaluation()
        except Exception as e:
            logger.error(f"Unexpected evaluating error: {e}")
            raise PipelineExecutionError(e) from e

    def run_pipeline(self):
        try:
            logger.info("Starting evaluating pipeline...")
            self._run_data_ingestion()
            self._train_model()
            self._evaluate_model()
            logger.info("Evaluating pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}")
            raise PipelineExecutionError(e) from e
