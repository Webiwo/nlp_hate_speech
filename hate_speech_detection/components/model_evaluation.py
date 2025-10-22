import shutil
import keras
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from hate_speech_detection.logger.logger import logger
from hate_speech_detection.exception.exception import ModelEvaluationError
from hate_speech_detection.entity.config_entity import (
    ModelEvaluationConfig,
    ModelTrainerConfig,
    DataTransformationConfig,
    PredictionConfig,
)
from hate_speech_detection.configuration.gcloud_syncer import GCloudSync
from sklearn.model_selection import train_test_split


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_config: DataTransformationConfig,
        prediction_config: PredictionConfig,
    ):
        self.eval_config = model_evaluation_config
        self.train_config = model_trainer_config
        self.trans_config = data_transformation_config
        self.pred_config = prediction_config
        self.gcloud = GCloudSync()

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

    def _evaluate(self, model):
        _, X_test, _, y_test = self._split_data()

        # X_test = pd.read_csv(self.train_config.x_test_path, encoding="utf-8")
        # y_test = pd.read_csv(self.train_config.y_test_path, encoding="utf-8")
        # X_test = X_test.astype(str).squeeze()
        # y_test = y_test.squeeze()

        with open(self.train_config.tokenizer_path, "rb") as f:
            load_tokenizer = pickle.load(f)

        X_test_vec = load_tokenizer(X_test)

        accuracy = model.evaluate(X_test_vec, y_test)
        logger.info(f"Loss & accuracy: {accuracy}")

        lstm_prediction = model.predict(X_test_vec)
        res = []
        for prediction in lstm_prediction:
            if prediction[0] < 0.5:
                res.append(0)
            else:
                res.append(1)
        logger.info(f"Confusion_matrix:\n{confusion_matrix(y_test,res)} ")
        return accuracy

    def _get_best_model_from_gcloud(self):
        self.gcloud.sync_folder_from_gcloud(
            self.eval_config.bucket_best_dir, self.eval_config.best_model_dir
        )

    def _push_best_model_to_gcloud(self):
        self.gcloud.sync_folder_to_gcloud(
            self.eval_config.bucket_best_dir, self.train_config.trained_model_dir
        )

    def _copy_prediction_artifacts(self, is_trained_model_accepted):
        if is_trained_model_accepted:
            logger.info(
                f"Copying {self.train_config.trained_model_path} to {self.pred_config.artifacts_dir}"
            )
            shutil.copy2(
                self.train_config.trained_model_path, self.pred_config.artifacts_dir
            )
        else:
            logger.info(
                f"Copying {self.eval_config.best_model_path} to {self.pred_config.artifacts_dir}"
            )
            shutil.copy2(
                self.eval_config.best_model_path, self.pred_config.artifacts_dir
            )
        logger.info(
            f"Copying {self.train_config.tokenizer_path} to {self.pred_config.artifacts_dir}"
        )
        shutil.copy2(self.train_config.tokenizer_path, self.pred_config.artifacts_dir)

    def initiate_model_evaluation(self):
        try:
            load_model = keras.models.load_model(self.train_config.trained_model_path)
            trained_accuracy = self._evaluate(model=load_model)
            is_trained_model_accepted = False
            self._get_best_model_from_gcloud()

            best_model_path = Path(self.eval_config.best_model_path)
            if best_model_path.exists() and best_model_path.is_file():
                best_model = keras.models.load_model(self.eval_config.best_model_path)
                best_model_accuracy = self._evaluate(best_model)

                if trained_accuracy[1] > best_model_accuracy[1]:
                    is_trained_model_accepted = True
                    self._push_best_model_to_gcloud()

            logger.info(f"Is trained model accepted: {is_trained_model_accepted}")
            self._copy_prediction_artifacts(is_trained_model_accepted)

        except Exception as e:
            raise ModelEvaluationError(e) from e
