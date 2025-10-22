from pathlib import Path
import re
import string
import emoji
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
from hate_speech_detection.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
)

from hate_speech_detection.exception.exception import DataTransformationError
from hate_speech_detection.logger.logger import logger


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_config: DataIngestionConfig,
    ):
        self.trans_config = data_transformation_config
        self.ingest_config = data_ingestion_config

    def _fix_broken_csv(self, input_path: str, output_path: str) -> None:
        """
        Fixes broken lines in a CSV file (e.g., from Twitter), merges
        emoji/UTF-8 fragments, and saves the fixed file to output_path
        """

        logger.info(f"Repairing a CSV file: {input_path}")
        fixed_lines = []
        buffer = ""

        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()

                # If the line looks like a fragment of an emoji or strange UTF-8 – merge with the previous one
                if re.match(r"^[ðâ\x80\x9f]+", stripped) and buffer:
                    buffer += " " + stripped
                    continue

                # if we have content in buffer, save it as a complete record
                if buffer:
                    fixed_lines.append(buffer)
                    buffer = ""

                # the new line becomes the current buffer
                buffer = stripped

        # save last record
        if buffer:
            fixed_lines.append(buffer)

        # save new CSV file (one tweet per line)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            for line in fixed_lines:
                f.write(line + "\n")

        logger.info(f"Repaired file saved: {output_path}")
        logger.info(f"Records after reconstruction: {len(fixed_lines)}")

    def _imbalanced_data_cleaning(self) -> pd.DataFrame:
        logger.info("Imbalanced data cleaning...")

        # input_path = self.ingest_config.imbalanced_data_path
        # fixed_path = str(Path(input_path).with_name("imbalanced_fixed.csv"))
        # self._fix_broken_csv(input_path, fixed_path)

        imbalanced_df = pd.read_csv(self.ingest_config.imbalanced_data_path)
        imbalanced_df = imbalanced_df[
            [self.trans_config.label_column, self.trans_config.tweet_column]
        ]
        return imbalanced_df

    def _raw_data_cleaning(self) -> pd.DataFrame:
        logger.info("Raw data cleaning...")

        # input_path = self.ingest_config.raw_data_path
        # fixed_path = str(Path(input_path).with_name("raw_fixed.csv"))
        # self._fix_broken_csv(input_path, fixed_path)

        raw_df = pd.read_csv(self.ingest_config.raw_data_path)
        raw_df = raw_df.drop(columns=self.trans_config.drop_columns, axis=1)
        class_column = self.trans_config.class_column
        label_column = self.trans_config.label_column
        raw_df.loc[raw_df[class_column] == 0, class_column] = 1
        raw_df.rename(columns={class_column: label_column}, inplace=True)
        raw_df.loc[raw_df[label_column] == 2, label_column] = 0
        return raw_df

    def _clean_and_concat_dataframes(self) -> pd.DataFrame:
        imbalanced_df = self._imbalanced_data_cleaning()
        raw_df = self._raw_data_cleaning()
        logger.info("Concatenating dataframes...")
        combined_df = pd.concat([imbalanced_df, raw_df], ignore_index=True)
        # combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        return combined_df

    def data_cleaning(self, words: str) -> str:
        language = self.trans_config.language
        stemmer = nltk.SnowballStemmer(language)
        stop_words = stopwords.words(language)
        more_stopwords = self.trans_config.more_stopwords
        stop_words += more_stopwords

        words = str(words).lower()
        words = re.sub(r"\[.*?\]", "", words)
        words = re.sub(r"https?://\S+|www\.\S+", "", words)
        words = re.sub(r"<.*?>+", "", words)
        words = re.sub(r"[%s]" % re.escape(string.punctuation), "", words)
        words = re.sub(r"\n", "", words)
        words = re.sub(r"\w*\d\w*", "", words)
        words = [word for word in words.split(" ") if word not in stop_words]
        words = " ".join(words)
        words = [stemmer.stem(words) for word in words.split(" ")]
        words = " ".join(words)
        return words

    def initiate_data_transformation(self):
        try:
            logger.info("Initiate data transformation...")
            df = self._clean_and_concat_dataframes()
            tweet = self.trans_config.tweet_column
            df[tweet] = df[tweet].apply(self.data_cleaning)
            df.to_csv(self.trans_config.transformed_file_path, encoding="utf-8")
        except Exception as e:
            raise DataTransformationError(e) from e
