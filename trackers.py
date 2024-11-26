import cv2
import numpy as np
from motpy import Detection, MultiObjectTracker
import logging

logger = logging.getLogger(__name__)

def count_square_circle(frame, original_image, config, contours, tracker_square, tracker_circle, unique_square_ids, unique_circle_ids, counted_square_ids, counted_circle_ids):
    circle_threshold = config.get('circle', 0.71)  # Tỷ lệ tối thiểu để xác định là hình tròn
    square_threshold = config.get('square', 0.86)  # Tỷ lệ tối thiểu để xác định là hình vuông

    list_square = []
    list_circle = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 235 and h > 235:  # Điều chỉnh ngưỡng nếu cần
            contour_area = cv2.contourArea(contour)
            bounding_box_area = w * h

            fill_ratio = contour_area / bounding_box_area
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if perimeter == 0:
                circularity = 0
            else:
                circularity = 4 * np.pi * (contour_area / (perimeter * perimeter))

            # Phân loại dựa trên tỷ lệ lấp đầy và độ tròn
            if fill_ratio > square_threshold and 4 <= len(approx) <= 8:
                detection = Detection(box=[x, y, x + w, y + h])
                list_square.append(detection)
            elif circularity > circle_threshold:
                detection = Detection(box=[x, y, x + w, y + h])
                list_circle.append(detection)

    tracker_square.step(detections=list_square)
    tracker_circle.step(detections=list_circle)

    active_tracked_squares = tracker_square.active_tracks(min_steps_alive=8, max_staleness=15)
    active_tracked_circles = tracker_circle.active_tracks(min_steps_alive=4, max_staleness=15)

    # Vẽ bounding box và ID cho hình vuông
    for obj in active_tracked_squares:
        x1, y1, x2, y2 = map(int, obj.box)
        square_id = f"SQUARE_{str(obj.id)[:5]}"  # Giới hạn ID còn 5 ký tự
        logger.info(f"Square ID: {square_id}, Bounding Box: ({x1}, {y1}, {x2}, {y2})")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Màu xanh lá cây cho hình vuông
        cv2.putText(frame, square_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if obj.id not in counted_square_ids:
            logger.info(f"New square detected: ID={square_id}")
            unique_square_ids.add(obj.id)
            counted_square_ids.add(obj.id)

    # Vẽ bounding box và ID cho hình tròn
    for obj in active_tracked_circles:
        x1, y1, x2, y2 = map(int, obj.box)
        circle_id = f"CIRCLE_{str(obj.id)[:5]}"  # Giới hạn ID còn 5 ký tự
        logger.info(f"Circle ID: {circle_id}, Bounding Box: ({x1}, {y1}, {x2}, {y2})")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Màu đỏ cho hình tròn
        cv2.putText(frame, circle_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if obj.id not in counted_circle_ids:
            logger.info(f"New circle detected: ID={circle_id}")
            unique_circle_ids.add(obj.id)
            counted_circle_ids.add(obj.id)
        
    # Cập nhật số đếm vào config để hiển thị
    square_count = len(unique_square_ids)
    circle_count = len(unique_circle_ids)

    # Hiển thị số lượng hình vuông và hình tròn
    cv2.putText(frame, f"Squares: {square_count}", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
    cv2.putText(frame, f"Circles: {circle_count}", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

    return frame
