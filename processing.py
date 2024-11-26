import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Mapping of threshold types
THRESHOLD_TYPES = {
    "THRESH_BINARY": cv2.THRESH_BINARY,
    "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
    "THRESH_TRUNC": cv2.THRESH_TRUNC,
    "THRESH_TOZERO": cv2.THRESH_TOZERO,
    "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV,
    "ADAPTIVE_THRESH_MEAN_C": cv2.ADAPTIVE_THRESH_MEAN_C,
    "ADAPTIVE_THRESH_GAUSSIAN_C": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    "THRESH_OTSU": cv2.THRESH_OTSU
}

def denoise(frame, config):
    c = config.get('denoise_kernel_size', 31)
    low_pass_kernel = np.ones((c, c), dtype=np.float32) / (c * c)
    frame_denoised = cv2.filter2D(frame, -1, low_pass_kernel)
    return frame_denoised

def threshold(frame, config):
    type_threshold = 0
    for item in config.get('type', []):
        type_threshold += THRESHOLD_TYPES.get(item, 0)
    low = config.get('low', 127)
    high = config.get('high', 255)
    arena_size = config.get('arena_size', 500)
    iterations_open = config.get('iterations_open', 2)
    iterations_close = config.get('iterations_close', 6)
    color = config.get('color', 255)
    thickness = config.get('thickness', 1)
    kernel_size = tuple(config.get('kernel_size', [5, 5]))
    kernel = np.ones(kernel_size, np.uint8)

    # Apply threshold
    ret, frame_threshold = cv2.threshold(frame, low, high, type_threshold)

    opened_image = cv2.erode(frame_threshold, kernel, iterations=iterations_open)
    closed_image = cv2.dilate(opened_image, kernel, iterations=iterations_close)

    # Find contours
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cleaned_image = np.zeros_like(closed_image)

    for contour in contours:
        if cv2.contourArea(contour) > arena_size:
            cv2.drawContours(cleaned_image, [contour], -1, color, thickness=thickness)

    cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2RGB)
    
    logger.info("Thresholding completed")
    
    return ret, cleaned_image, frame_threshold, contours

def make_frame_show(frames=[], frame_size=(640, 480), required_frames=4):
    if required_frames == 1:
        if len(frames) == 0:
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        else:
            frame = frames[0]
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame = cv2.resize(frame, frame_size)
        return frame
    else:
        while len(frames) < required_frames:
            blank_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            frames.append(blank_frame)
    
        for i in range(len(frames)):
            if len(frames[i].shape) == 2 or frames[i].shape[2] == 1:
                frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2RGB)
            frames[i] = cv2.resize(frames[i], frame_size)
        
        if required_frames == 4:
            top_row = np.hstack((frames[0], frames[1]))
            bottom_row = np.hstack((frames[2], frames[3]))
            combined_frame = np.vstack((top_row, bottom_row))
        elif required_frames == 2:
            combined_frame = np.hstack((frames[0], frames[1]))
        else:
            combined_frame = frames[0]  # Default to the first frame
        
        return combined_frame


def create_final_frame(total_squares, total_circles, frame_size=(640, 480)):
    final_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)  # Black frame

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color_square = (0, 255, 0)  # Green
    color_circle = (0, 0, 255)  # Red
    thickness = 3

    # Calculate text size to center it
    text_square = f"Total Squares: {total_squares}"
    text_circle = f"Total Circles: {total_circles}"

    # Get text size
    (text_width_sq, text_height_sq), _ = cv2.getTextSize(text_square, font, font_scale, thickness)
    (text_width_ci, text_height_ci), _ = cv2.getTextSize(text_circle, font, font_scale, thickness)

    # Calculate positions
    pos_square = ((frame_size[0] - text_width_sq) // 2, (frame_size[1] // 2) - 30)
    pos_circle = ((frame_size[0] - text_width_ci) // 2, (frame_size[1] // 2) + 30)

    # Put text on the final frame
    cv2.putText(final_frame, text_square, pos_square, font, font_scale, color_square, thickness, cv2.LINE_AA)
    cv2.putText(final_frame, text_circle, pos_circle, font, font_scale, color_circle, thickness, cv2.LINE_AA)

    return final_frame
