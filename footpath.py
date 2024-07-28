import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag
import math

def fit_line_and_display(points):
    points = np.array(points, dtype=np.float32).reshape((-1, 1, 2))

    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    slope = vy / vx
    angle_radians = math.atan(slope)
    angle_degrees = math.degrees(angle_radians)

    x0 = 0
    y0 = int(y - (x * slope))
    x1 = 500
    y1 = int(y + ((500 - x) * slope))

    return (x0, y0), (x1, y1), angle_degrees


def find_best_split(frame):
    total_ones = np.sum(frame)
    min_diff = float('inf')
    best_index = 0
    best_split = 'vertical'
    
    left_ones = 0
    for col in range(frame.shape[1]):
        left_ones += np.sum(frame[:, col])
        right_ones = total_ones - left_ones
        diff = abs(left_ones - right_ones)
        if diff < min_diff:
            min_diff = diff
            best_index = col
            best_split = 'vertical'
    
    top_ones = 0
    for row in range(frame.shape[0]):
        top_ones += np.sum(frame[row, :])
        bottom_ones = total_ones - top_ones
        diff = abs(top_ones - bottom_ones)
        if diff < min_diff:
            min_diff = diff
            best_index = row
            best_split = 'horizontal'

    return best_split, best_index


def apriltag_detection(gray_image):
    options = apriltag.DetectorOptions(families="tag16h5")
    detector = apriltag.Detector(options)

    detections = detector.detect(gray_image)

    left_shoe_position = []
    right_shoe_position = []

    for detection in detections:
        tag_id = detection.tag_id

        if tag_id == 1:
            left_shoe_position = [
                np.array(detection.corners[1], dtype=int),
                np.array(detection.corners[2], dtype=int),
            ]
        elif tag_id == 2:
            right_shoe_position = [
                np.array(detection.corners[0], dtype=int),
                np.array(detection.corners[3], dtype=int),
            ]

    return left_shoe_position, right_shoe_position


def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    threshold_value = 150
    _, light_mask = cv2.threshold(v_channel, threshold_value, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(v_channel, threshold_value, 255, cv2.THRESH_BINARY_INV)

    light_colors = cv2.bitwise_and(frame, frame, mask=light_mask)
    dark_colors = cv2.bitwise_and(frame, frame, mask=dark_mask)

    binary_map = (v_channel >= threshold_value).astype(np.uint8) * 255

    return light_colors, dark_colors, binary_map


def process_video(input_source, source_type="video"):
    if source_type == "video":
        cap = cv2.VideoCapture(input_source)
    elif source_type == "realsense":
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
    left_shoe_points, right_shoe_points = [], []
    while True:
        if source_type == "video":
            ret, frame = cap.read()
            if not ret:
                break
        elif source_type == "realsense":
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, p2 = apriltag_detection(gray)

        if len(p1) == 0 and len(p2) == 0: 
            continue
        if len(p1) != 0:
            left_shoe_points = p1
        if len(p2) != 0:
            right_shoe_points = p2
        if len(left_shoe_points) == 0 or len(right_shoe_points) == 0:
            continue

        shoe_p1 = (
            (left_shoe_points[0][0] + right_shoe_points[0][0]) // 2,
            (left_shoe_points[0][1] + right_shoe_points[0][1]) // 2,
        )
        shoe_p2 = (
            (left_shoe_points[1][0] + right_shoe_points[1][0]) // 2,
            (left_shoe_points[1][1] + right_shoe_points[1][1]) // 2,
        )
        copy_frame = frame
        
        cv2.circle(copy_frame, shoe_p1, 5, (0, 0, 255), -1)


        step = 30
        try:
            points = [shoe_p1]
            for n in range(0, shoe_p1[1] - step, step):
                light_colors, dark_colors, binary_frame = process_frame(frame[n:n+step,:])
                
                # Get the coordinates of the 255 values
                y_coords, x_coords = np.where(binary_frame == 255)

                # Calculate the x-coordinate that separates the image into two equal areas
                # This is done by finding the median x-coordinate
                median_x = np.median(x_coords)

                # Define the line endpoints
                height = binary_frame.shape[0]
                x1, x2 = int(median_x), int(median_x)
                y1, y2 = 0, height
                
                # Calculate the middle of the vertical line
                middle_x = int(median_x)
                middle_y = height // 2
                # Draw the vertical line on the color image
                # cv2.line(copy_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(copy_frame, (middle_x, middle_y + n), 5, (0, 255, 0), 2)
                
                if n // step < 3:
                    points.append((middle_x, middle_y + n))
                        
            d1, d2, angle_degree = fit_line_and_display(points)
            cv2.line(copy_frame, d1, d2, (255, 0, 0), 2)

            if angle_degree < 0:
                print("Right")
            elif angle_degree < 60:
                print("Left")

        except:
            print("no")
            
        # Display the image with the separator line
        cv2.imshow('Separator Line', copy_frame)
        # cv2.imshow("Frame", frame)
        # cv2.imshow("blur", blur_image)
        # cv2.imshow('Light Colors', light_colors)
        # cv2.imshow('Dark Colors', dar_colors)
        cv2.imshow('Binary Map', binary_frame)

        # Exit if the user presses the 'q' key
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    if source_type == "video":
        cap.release()
    elif source_type == "realsense":
        pipeline.stop()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_source = "april.mp4"  # Change this to your video file path or to 'realsense' to use the RealSense camera
    source_type = "video"  # Set this to 'video' for a video file or 'realsense' for RealSense camera
    # source_type = 'realsense'  # Set this to 'video' for a video file or 'realsense' for RealSense camera

    process_video(input_source, source_type)