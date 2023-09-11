import os
import sys
import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from src.exception import CustomException
from src.logger import CustomLogger


# nltk.download('punkt')

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.logger = CustomLogger().get_logger()

    def preprocess_text(self, text):
        # Tokenization and preprocessing steps
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove non-alphabetic characters
        tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        print(tokens)
        return ' '.join(tokens)

    def get_data_transformation_obj(self):
        # Create and configure TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,  # You can adjust max_features as needed
            tokenizer=word_tokenize,
            stop_words=stopwords.words('english'),
            lowercase=True
        )
        return vectorizer

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            self.logger.info("Read train and test data is completed")
            self.logger.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_obj()

            target_column_name = "label"

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            self.logger.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df['text'])
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df['text'])

            train_arc = np.c_[input_feature_train_arr.toarray(), np.array(target_feature_train_df)]  # Concatenate arrays
            test_arc = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df)]

            self.logger.info("Saved preprocessing Object.")

            # Save the preprocessor object using pickle
            with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as file:
                pickle.dump(preprocessing_obj, file)

            return train_arc, test_arc, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == '__main__':
#     # Replace 'train_data.csv' and 'test_data.csv' with the actual file paths.
#     train_path = r'C:\Users\kishu\PycharmProjects\detecting-fake-news\src\components\artifacts\train.csv'
#     test_path = r'C:\Users\kishu\PycharmProjects\detecting-fake-news\src\components\artifacts\test.csv'
#
#     data_transformation = DataTransformation()
#     train_arc, test_arc, preprocessor_obj_path = data_transformation.initiate_data_transformation(train_path, test_path)
#
#     # Print or log the results as needed
#     print("Training data shape:", train_arc.shape)
#     print("Testing data shape:", test_arc.shape)
#     print("Preprocessor object saved at:", preprocessor_obj_path)

