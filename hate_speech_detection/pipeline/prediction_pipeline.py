import keras
import pickle
from hate_speech_detection.configuration.config_manager import ConfigurationManager
from hate_speech_detection.exception.exception import PipelineExecutionError
from hate_speech_detection.logger.logger import logger
from hate_speech_detection.components.data_transforamation import DataTransformation


class PredictionPipeline:

    def __init__(self, configuration_manager=None):
        self.config_manager = configuration_manager or ConfigurationManager()
        self.pred_config = self.config_manager.get_prediction_config()
        self.trans_config = self.config_manager.get_data_transformation_config()
        self.ingest_config = self.config_manager.get_data_ingestion_config()
        self.data_transform = DataTransformation(self.trans_config, self.ingest_config)

    def _predict(self, text):
        load_model = keras.models.load_model(self.pred_config.model_path)
        with open(self.pred_config.tokenizer_path, "rb") as f:
            load_tokenizer = pickle.load(f)

        text = self.data_transform.data_cleaning(text)
        text = [text]
        logger.info(f"TEXT::: {text}")

        text_vec = load_tokenizer(text)
        pred = load_model.predict(text_vec)

        if pred > 0.5:
            logger.info("hate and abusive")
            return "hate"
        else:
            logger.info("no hate")
            return "no hate"

    def run_pipeline(self, text):
        try:
            logger.info("Starting prediction pipeline...")
            result = self._predict(text)
            logger.info("Prediction pipeline completed successfully.")
            return result

        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}")
            raise PipelineExecutionError(e) from e
