import pickle
import io
from sre_parse import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from hate_speech_detection.components.data_tokenizer import DataTokenizer
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
        df = df.dropna()
        X = df[self.trans_config.tweet_column]
        y = df[self.trans_config.label_column]

        logger.info(f"Data cardinality (X,y) : ({len(X)}, {len(y)})")

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

    def initiate_model_trainer(self):
        try:
            logger.info("Model training started...")
            X_train, X_test, y_train, y_test = self._split_data()

            X_train.to_csv(
                self.train_config.x_train_path, index=False, encoding="utf-8"
            )
            y_train.to_csv(
                self.train_config.y_train_path, index=False, encoding="utf-8"
            )
            X_test.to_csv(self.train_config.x_test_path, index=False, encoding="utf-8")
            y_test.to_csv(self.train_config.y_test_path, index=False, encoding="utf-8")

            tokenizer = DataTokenizer(self.train_config)
            X_train, vectorizer = tokenizer.tokenize(X_train)

            model_architecture = ModelArchitecture(self.train_config)
            model = model_architecture.get_model()

            model.fit(
                X_train,
                y_train,
                batch_size=self.train_config.batch_size,
                epochs=self.train_config.epochs,
                validation_split=self.train_config.test_split,
            )

            model.summary(print_fn=lambda x: logger.info(x))

            model.save(self.train_config.trained_model_path)

            with io.open(self.train_config.tokenizer_path, "wb") as f:
                pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info("Model training finished...")
        except Exception as e:
            raise ModelTrainingError(e) from e
