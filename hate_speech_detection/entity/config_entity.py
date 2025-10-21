from dataclasses import dataclass
import token
import tokenize


@dataclass
class DataIngestionConfig:
    bucket_name: str
    zip_file_name: str
    artifacts_dir: str
    imbalanced_data_path: str
    raw_data_path: str


@dataclass
class DataTransformationConfig:
    artifacts_dir: str
    transformed_file_path: str
    drop_columns: list
    class_column: str
    label_column: str
    tweet_column: str
    language: str
    more_stopwords: list


@dataclass
class ModelTrainerConfig:
    artifacts_dir: str
    tokenizer_path: str
    trained_model_dir: str
    trained_model_path: str
    x_train_path: str
    y_train_path: str
    x_test_path: str
    y_test_path: str
    random_state: int
    epochs: int
    batch_size: int
    test_split: float
    max_words: int
    max_len: int
    loss: str
    metrics: list
    activation: str


@dataclass
class ModelEvaluationConfig:
    artifacts_dir: str
    bucket_best_dir: str
    best_model_dir: str
    best_model_path: str


@dataclass
class ModelPusherConfig:
    artifacts_dir: str
    app_host: str
    app_port: int
