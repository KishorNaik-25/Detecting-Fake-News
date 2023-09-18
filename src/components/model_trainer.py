import os
import sys
import pickle
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC  # You can choose a different classifier
from src.exception import CustomException
from src.logger import CustomLogger
from src.components.Data_Tranformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.logger = CustomLogger().get_logger()

    def save_trained_model(self, trained_model):
        try:
            # Save the trained model to the specified file path
            with open(self.model_trainer_config.trained_model_file_path, 'wb') as model_file:
                pickle.dump(trained_model, model_file)

            print(f"Trained model saved as {self.model_trainer_config.trained_model_file_path}")
            self.logger.info("The model is saved")

        except Exception as e:
            raise CustomException(e, sys)

    def load_data_and_preprocessor(self, train_path, test_path):
        try:
            data_transformation = DataTransformation()
            train_arc, test_arc, preprocessor_obj_path = data_transformation.initiate_data_transformation(train_path, test_path)
            return train_arc, test_arc, preprocessor_obj_path

        except Exception as e:
            raise CustomException(e, sys)

    def train_and_evaluate_model(self, train_arc, test_arc):
        try:
            X_train = train_arc[:, :-1]  # Features
            y_train = train_arc[:, -1]   # Labels

            X_test = test_arc[:, :-1]    # Features
            y_test = test_arc[:, -1]     # Labels

            # Initialize and train the classifier (you can choose a different classifier)
            classifier = SVC(kernel='linear')
            classifier.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = classifier.predict(X_test)

            # Calculate accuracy and print classification report
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Save the trained classifier immediately upon training
            self.save_trained_model(classifier)

            return accuracy, classification_rep, conf_matrix , classifier  # Return the trained classifier

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    # Replace with the actual file paths for the train and test data
    train_path = r'C:\Users\kishu\PycharmProjects\detecting-fake-news\src\pipeline\artifacts\train.csv'
    test_path = r'C:\Users\kishu\PycharmProjects\detecting-fake-news\src\pipeline\artifacts\test.csv'

    # Create an instance of ModelTrainer
    model_trainer = ModelTrainer()

    # Load data and preprocessing object
    train_arc, test_arc, preprocessing_obj_path = model_trainer.load_data_and_preprocessor(train_path, test_path)

    # Train and evaluate the model, and get the trained classifier
    accuracy, classification_rep, trained_classifier = model_trainer.train_and_evaluate_model(train_arc, test_arc)

    # Save the trained classifier using the ModelTrainer instance
    model_trainer.save_trained_model(trained_classifier)

    # Print or log the results as needed
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)