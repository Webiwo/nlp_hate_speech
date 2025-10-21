import tensorflow as tf
from keras.layers import TextVectorization
from hate_speech_detection.entity.config_entity import ModelTrainerConfig
from hate_speech_detection.logger.logger import logger


class DataTokenizer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.train_config = model_trainer_config

    def tokenize(self, X_data):
        logger.info("Tokenizing text data...")
        vectorize_layer = TextVectorization(
            max_tokens=self.train_config.max_words,
            output_mode="int",
            output_sequence_length=self.train_config.max_len,
            ngrams=(1, 4),
        )
        ds = tf.data.Dataset.from_tensor_slices(X_data).batch(
            self.train_config.batch_size
        )
        vectorize_layer.adapt(ds)
        X_data_vectorized = vectorize_layer(X_data)

        logger.info("Text data tokenization completed.")
        return X_data_vectorized, vectorize_layer
