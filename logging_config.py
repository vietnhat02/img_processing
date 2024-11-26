import logging
import os

def setup_logging():
    os.environ['MOTPY_LOG_LEVEL'] = 'DEBUG'
    # Cấu hình logger gốc
    logging.basicConfig(
        level=logging.DEBUG,  # Đặt thành DEBUG để thu thập tất cả các log
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),               # Log hiện console
            logging.FileHandler('process_log.log')     # Log vào file 'app_log.log'
        ]
    )

    # Điều chỉnh tất cả các logger của motpy
    motpy_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('motpy')]

    for logger_motpy in motpy_loggers:
        logger_motpy.setLevel(logging.DEBUG)
        logger_motpy.propagate = True
        # Loại bỏ NullHandler nếu có
        handlers_to_remove = [handler for handler in logger_motpy.handlers if isinstance(handler, logging.NullHandler)]
        for handler in handlers_to_remove:
            logger_motpy.removeHandler(handler)
