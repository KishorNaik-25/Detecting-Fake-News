from src.components.Data_Ingestion import DataIngestion
from src.components.Data_Tranformation import DataTransformation

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arc, test_arc, preprocessor_obj_path = data_transformation.initiate_data_transformation(train_data, test_data)

