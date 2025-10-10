from dataclasses import dataclass


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
    transformed_file_name: str
    drop_columns: list
    class_column: str
    label_column: str
    tweet_column: str


@dataclass
class ModelTrainerConfig:
    artifacts_dir: str
    trained_model_name: str
    x_test_file: str
    y_test_file: str
    x_train_file: str
    random_state: int
    epochs: int
    batch_size: int
    validation_split: float
    max_words: int
    max_len: int
    loss: str
    metrics: list
    activation: str
