import pandas as pd
from hate_speech_detection.exception.exception import DataValidationError
from hate_speech_detection.logger.logger import logger


class DataValidator:
    """
    Validates input CSV data for:
      - required columns,
      - missing values in columns,
      - correct data types.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        if "raw" in file_path.lower():
            self.required_columns = [
                "Unnamed: 0",
                "count",
                "hate_speech",
                "offensive_language",
                "neither",
                "class",
                "tweet",
            ]
            self.int_cols = [
                "count",
                "hate_speech",
                "offensive_language",
                "neither",
                "class",
            ]
        elif "imbalanced" in file_path.lower():
            self.required_columns = ["id", "label", "tweet"]
            self.int_cols = ["id", "label"]
        else:
            raise DataValidationError("Unknown CSV type based on filename.")

    def validate(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            raise DataValidationError(f"CSV loading error: {e}") from e

        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        # Check nulls in key columns
        nulls = df[self.required_columns].isnull().sum()
        if nulls.sum() > 0:
            raise DataValidationError(
                f"Missing values in key columns: {nulls.to_dict()}"
            )

        # Check integer columns
        for col in self.int_cols:
            if not pd.api.types.is_integer_dtype(df[col]):
                raise DataValidationError(f"Column {col} is not of integer type.")

        logger.info(f"CSV validation succeeded: {self.file_path}")
        return df
