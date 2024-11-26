import argparse
import yaml
import cv2
import logging
from screeninfo import get_monitors

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Video processing with configurable options.")

    parser.add_argument("--video_path", type=str, help="Đường dẫn video input")
    parser.add_argument("--name", type=str, help="Tên")
    parser.add_argument("--mssv", type=str, help="Mã số")
    parser.add_argument("--low_threshold", type=int, help="Lọc nhiễu nền low_threshold")
    parser.add_argument("--denoise_c", type=int, help="Thông số giảm nhiễu")
    parser.add_argument("--iterations_open", type=int, help="Số lần xói mòn (erode)")
    parser.add_argument("--iterations_close", type=int, help="Số lần giãn nở (dilate)")

    return parser.parse_args()

def load_config(config_path="config.yaml"):
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        logger.debug("Configuration loaded")
    return data

def get_screen_resolution():
    monitor = get_monitors()[0]  # Get the first monitor
    return (monitor.width, monitor.height)

def put_text(frame, config):
    # Top-left corner texts
    cv2.putText(
        frame, 
        config.get('mssv', 'MSSV'), 
        (10, 60), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2, 
        cv2.LINE_AA
    )
    cv2.putText(
        frame, 
        config.get('name', 'Name'), 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2, 
        cv2.LINE_AA
    )
    
