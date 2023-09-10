import logging
import sys

def get_error_message_details(error, error_detail):
    exc_type, exc_obj, exc_tb = error_detail
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in Python script: {file_name}, line number: {exc_tb.tb_lineno}, error message: {error}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = get_error_message_details(error_message, error_detail)
        self.error_detail = error_detail

    def __str__(self):
        return self.error_message

# if __name__ == "__main__":
#     try:
#         # Simulate an error
#         1 / 0
#     except ZeroDivisionError as error:
#         try:
#             raise CustomException("Custom error message", sys.exc_info()) from None
#         except CustomException as custom_error:
#             print(custom_error)
