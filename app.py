import sys
import os
from hate_speech_detection.configuration.config_manager import ConfigurationManager
from hate_speech_detection.pipeline.train_pipeline import TrainPipeline
from hate_speech_detection.pipeline.prediction_pipeline import PredictionPipeline
from hate_speech_detection.logger.logger import logger
from hate_speech_detection.exception.exception import PipelineExecutionError


if __name__ == "__main__":
    os.environ["PYTHONUTF8"] = "1"

    try:
        config_manager = ConfigurationManager()

        train_pipeline = TrainPipeline(config_manager)
        train_pipeline.run_pipeline()

        pred_pipeline = PredictionPipeline(config_manager)
        pred_pipeline.run_pipeline(
            "He is soooo stupid. I'll catch that bastard tomorrow!!!"
        )

    except PipelineExecutionError as e:
        logger.error(f"Pipeline terminated with error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unhandled exception occurred")
        sys.exit(1)
