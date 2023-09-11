from src.components.Data_Ingestion import DataIngestion

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
