
"""
Logging python file script
"""

import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format = logging_str,

    handlers=[
        logging.FileHandler(log_filepath), # create a log folder and save all the all logging
        logging.StreamHandler(sys.stdout)  # printing log in the terminal
    ]
)

logger = logging.getLogger("Mlprojectlogger")