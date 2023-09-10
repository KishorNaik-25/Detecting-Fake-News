import logging
import os
from datetime import datetime

class CustomLogger:
    def __init__(self):
        self.LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        logs_path = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_path, exist_ok=True)
        self.LOG_FILE_PATH = os.path.join(logs_path, self.LOG_FILE)

        self.logger = logging.getLogger("my_custom_logger")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(self.LOG_FILE_PATH)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

if __name__ == "__main__":
    custom_logger = CustomLogger()
    logger = custom_logger.get_logger()

    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
