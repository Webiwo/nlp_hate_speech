# Creating model architecture.
from hate_speech_detection.entity.config_entity import ModelTrainerConfig
from hate_speech_detection.logger.logger import logger
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import (
    LSTM,
    Dense,
    Embedding,
    SpatialDropout1D,
)


class ModelArchitecture:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def get_model(self) -> Sequential:
        model = Sequential()
        model.add(
            Embedding(self.config.max_words, 100, input_length=self.config.max_len)
        )
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation=self.config.activation))
        model.compile(
            loss=self.config.loss, optimizer=RMSprop(), metrics=self.config.metrics
        )

        logger.info("Model architecture created successfully.")
        return model
