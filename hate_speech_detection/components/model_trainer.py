from logging import config
from math import exp
import os
from pyexpat import model
import sys
import pickle
from turtle import mode
import pandas as pd
import proto
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization
from hate_speech_detection.logger.logger import logger
from hate_speech_detection.exception.exception import ModelTrainingError
from hate_speech_detection.entity.config_entity import (
    ModelTrainerConfig,
    DataTransformationConfig,
)
from hate_speech_detection.ml.model import ModelArchitecture


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_config: DataTransformationConfig,
    ):
        self.train_config = model_trainer_config
        self.trans_config = data_transformation_config

    def _split_data(self):
        logger.info("Splitting data into train and test sets...")
        df = pd.read_csv(self.trans_config.transformed_file_path)
        X = df[self.trans_config.tweet_column]
        y = df[self.trans_config.label_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.train_config.test_split,
            random_state=self.train_config.random_state,
        )
        logger.info(
            f"Data splitting completed - X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    def _topkenize(self, X_train, X_test):
        logger.info("Tokenizing text data...")
        vectorize_layer = TextVectorization(
            max_tokens=self.train_config.max_words,
            output_mode="int",
            output_sequence_length=self.train_config.max_len,
            ngrams=(1, 4),
        )
        ds = tf.data.Dataset.from_tensor_slices(X_train).batch(
            self.train_config.batch_size
        )
        vectorize_layer.adapt(ds)
        X_train_vectorized = vectorize_layer(X_train)
        X_test_vectorized = vectorize_layer(X_test)

        logger.info("Text data tokenization completed.")
        return X_train_vectorized, X_test_vectorized, vectorize_layer

    def initiate_model_trainer(self, pickle=False):
        try:
            logger.info("Model training started...")
            X_train, X_test, y_train, y_test = self._split_data()
            X_train = X_train.dropna()
            X_test = X_test.dropna()

            X_train.to_csv(self.train_config.x_train_path, index=False)
            y_train.to_csv(self.train_config.y_train_path, index=False)
            X_test.to_csv(self.train_config.x_test_path, index=False)
            y_test.to_csv(self.train_config.y_test_path, index=False)

            X_train, X_test, vectorizer = self._topkenize(X_train, X_test)

            model_architecture = ModelArchitecture(self.train_config)
            model = model_architecture.get_model()

            history = model.fit(
                X_train,
                y_train,
                batch_size=self.train_config.batch_size,
                epochs=self.train_config.epochs,
                validation_split=self.train_config.test_split,
            )

            model.save(self.train_config.trained_model_path)

            if pickle:
                with open(self.train_config.tokenizer_path, "wb") as f:
                    pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                vectorizer_model = tf.keras.models.Sequential([vectorizer])
                vectorizer_model.save(
                    self.train_config.tokenizer_path.replace(".pickle", ".keras")
                )

            logger.info("Model training finished...")
            logger.info(history)
        except Exception as e:
            raise ModelTrainingError(e) from e
