import sys
from hate_speech_detection.pipeline.train_pipeline import TrainPipeline
from hate_speech_detection.logger.logger import logger
from hate_speech_detection.exception.exception import PipelineExecutionError


if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
    except PipelineExecutionError as e:
        logger.error(f"Pipeline terminated with error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unhandled exception occurred")
        sys.exit(1)
