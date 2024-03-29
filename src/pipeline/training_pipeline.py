from src.components.Data_Ingestion import DataIngestion
from src.components.Data_Tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arc, test_arc, preprocessor_obj_path = data_transformation.initiate_data_transformation(train_data, test_data)

    model_train = ModelTrainer()
    accuracy, classification_rep, conf_matrix,_ = model_train.train_and_evaluate_model(train_arc, test_arc)

     # Display accuracy and classification report
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)
    print("Confusion matrix:\n",conf_matrix)







