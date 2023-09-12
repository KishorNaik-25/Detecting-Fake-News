import os
import sys
import pickle
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.exception import CustomException
from src.logger import CustomLogger
import numpy as np

@dataclass
class ModelPredictorConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class ModelPredictor:
    def __init__(self):
        self.model_predictor_config = ModelPredictorConfig()
        self.logger = CustomLogger().get_logger()

    def load_trained_model(self):
        try:
            with open(self.model_predictor_config.trained_model_file_path, 'rb') as model_file:
                trained_model = pickle.load(model_file)
            return trained_model
        except Exception as e:
            raise CustomException(e, sys)

    def load_preprocessor(self):
        try:
            with open(self.model_predictor_config.preprocessor_obj_file_path, 'rb') as preprocessor_file:
                preprocessor = pickle.load(preprocessor_file)
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_text(self, text, preprocessor):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        processed_text = ' '.join(tokens)
        return preprocessor.transform([processed_text])

    def predict(self, text):
        try:
            trained_model = self.load_trained_model()
            preprocessor = self.load_preprocessor()
            processed_text = self.preprocess_text(text, preprocessor)

            # Convert the sparse input to dense format
            processed_text_dense = processed_text.toarray()

            prediction = trained_model.predict(processed_text_dense)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    model_predictor = ModelPredictor()
    input_text = """"Will Trump pull a Brexit times ten? What would it take, beyond WikiLeaks, to bring the Clinton (cash) machine down? Will Hillary win and then declare WWIII against her Russia/Iran/Syria “axis of evil”? Will the Middle East totally explode? Will the pivot to Asia totally implode? Will China be ruling the world by 2025?"""
    prediction = model_predictor.predict(input_text)
    print("Predicted Label:", prediction)
