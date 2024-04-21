import cv2
import numpy as np
import math
import pyttsx3

FORWARD = 0
ROTATE_RIGHT = 1
ROTATE_LEFT = 2
MOVE_RIGHT = 3
MOVE_LEFT = 4

class ImageProcessor:
    def __init__(self):
        self.x_middle, self.x_middle_next, self.y_middle, self.y_middle_next = 0, 0, 0, 0
        self.gap_1 = 120
        self.gap_2 = 240
        self.state = FORWARD
        self.counter = 0
        self.r_middle, self.theta_middle = 0, 0


    def polar_to_cartesian(rho, theta):
        x = rho * math.cos(theta)
        y = rho * math.sin(theta)
        return x, y
    
    def cartesian_to_polar(x, y):
        rho = math.sqrt(x**2 + y**2)
        theta = math.atan2(y, x)
        return rho, theta

    def find_polar_coordinates(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1

        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        return r, theta

    def process_video(self, video_file):
        cap = cv2.VideoCapture(video_file)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Thresholding
            thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

            # Morphological Operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            eroded = cv2.erode(thresholded, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)

            # Contour Detection
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Shoe Detection
            local_min_x = float('inf')
            local_min_y = float('inf')
            local_max_x = float('-inf')
            local_max_y = float('-inf')
            local_shoe_updated = False
            c_c = 0 #contour counter
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Adjust the area threshold as needed
                    # Check color similarity (assuming black shoes)
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = frame[y:y+h, x:x+w]
                    avg_color = np.average(roi, axis=(0, 1))
                    if avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:  # Adjust color threshold as needed
                        # Update the minimum and maximum coordinates
                        local_min_x = min(local_min_x, x)
                        local_min_y = min(local_min_y, y)
                        local_max_x = max(local_max_x, x + w)
                        local_max_y = max(local_max_y, y + h)
                        local_shoe_updated = True
                        c_c += 1
                        self.gap_1 = max(local_min_y - 50, 0)
                        self.gap_2 = max(local_min_y, 0)

                        if local_min_y == float('inf'):
                            self.gap_1 = 120
                            self.gap_2 = 240
            
            blur_image = blurred[self.gap_1:self.gap_2]

            # Apply Canny edge detection
            edges = cv2.Canny(blur_image, 70, 120)

            # Apply Hough Transform for line detection
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=30)

            # Draw the detected lines on the original image
            result_image = frame.copy()
            parallel_lines = []
            if lines is not None:
                for idx_1 in range(lines.shape[0]):
                    rho_1, theta_1 = lines[idx_1, 0]
                    # a_1 = np.cos(theta_1)
                    # b_1 = np.sin(theta_1)
                    # x0_1 = a_1 * rho_1
                    # y0_1 = b_1 * rho_1
                    # x1_1 = int(x0_1 + 1000 * (-b_1))
                    # y1_1 = int(y0_1 + 1000 * (a_1))
                    # x2_1 = int(x0_1 - 1000 * (-b_1))
                    # y2_1 = int(y0_1 - 1000 * (a_1))

                    # if abs(x2_1 - x1_1) < 1e-5:
                    #     continue

                    for idx_2 in range(idx_1 + 1, lines.shape[0]):
                        rho_2, theta_2 = lines[idx_2, 0]
                        # a_2 = np.cos(theta_2)
                        # b_2 = np.sin(theta_2)
                        # x0_2 = a_2 * rho_2
                        # y0_2 = b_2 * rho_2
                        # x1_2 = int(x0_2 + 1000 * (-b_2))
                        # y1_2 = int(y0_2 + 1000 * (a_2))
                        # x2_2 = int(x0_2 - 1000 * (-b_2))
                        # y2_2 = int(y0_2 - 1000 * (a_2))

                        # if abs(x2_2 - x1_2) < 1e-5:
                        #     continue

                        if abs(theta_1 - theta_2) < 0.2:
                            if abs(rho_1 - rho_2) > 150:
                                # parallel_lines.append((x1_1, y1_1, x2_1, y2_1))
                                # parallel_lines.append((x1_2, y1_2, x2_2, y2_2))
                                parallel_lines.append((rho_1, theta_1))
                                parallel_lines.append((rho_2, theta_2))
                                break

                    if len(parallel_lines) == 2:
                        break

            # Create a separate image for parallel lines and middle line
            parallel_image = frame.copy()
            
            # new_line = True

            # Draw the parallel lines on the separate image
            if len(parallel_lines) == 2:
                # x1, y1, x2, y2 = parallel_lines[0]
                # x1_next, y1_next, x2_next, y2_next = parallel_lines[1]
                # cv2.line(parallel_image, (x1, y1+self.gap_1), (x2, y2+self.gap_1), (0, 255, 0), 2)
                # cv2.line(parallel_image, (x1_next, y1_next+self.gap_1), (x2_next, y2_next+self.gap_1), (0, 255, 0), 2)

                # # Calculate the middle line coordinates
                # self.x_middle = (x1 + x1_next) // 2
                # self.y_middle = (y1 + y1_next) // 2
                # self.x_middle_next = (x2 + x2_next) // 2
                # self.y_middle_next = (y2 + y2_next) // 2
                rho_1, theta_1 = parallel_lines[0]
                rho_2, theta_2 = parallel_lines[1]
                
                x1, y1 = ImageProcessor.polar_to_cartesian(rho_1, theta_1)
                x2, y2 = ImageProcessor.polar_to_cartesian(rho_2, theta_2)

                self.x_middle = min(x1, x2) + (abs(x1 - x2) // 2)
                self.y_middle = min(y1, y2) + (abs(y1 - y2) // 2)

                self.r_middle, self.theta_middle = ImageProcessor.cartesian_to_polar(self.x_middle, self.y_middle)
                # Draw the middle line in blue color
                # cv2.line(parallel_image, (self.x_middle, self.y_middle+self.gap_1), (self.x_middle_next, self.y_middle_next+self.gap_1), (255, 0, 0), 2)
                # new_line = False
            # if new_line:
                # cv2.line(parallel_image, (self.x_middle, self.y_middle+self.gap_1), (self.x_middle_next, self.y_middle_next+self.gap_1), (255, 0, 0), 2)

            # reset state after 50 frames
            if self.counter == 50:
                self.state = FORWARD
                self.counter = 0
            self.counter += 1

            # feedback
            if local_shoe_updated and c_c == 2: # if the local shoe is change and founded contours are equl to 2

                # convert point to polar coordinate for better comparison
                r_shoe = (local_max_x + local_min_x) // 2

                # r_shoe, theta_shoe = ImageProcessor.find_polar_coordinates(mid_shoe, local_min_y, mid_shoe, local_max_y)
                # r_middle, theta_middle = ImageProcessor.find_polar_coordinates(self.x_middle, self.y_middle, self.x_middle_next, self.y_middle_next)

                # check if lines are parallel
                # check for off-center
                if self.theta_middle > 0.2:
                    if self.state != ROTATE_RIGHT:
                        self.state = ROTATE_RIGHT
                        print("rotate_right")

                elif self.theta_middle < -0.2:
                    if self.state != ROTATE_LEFT:
                        self.state = ROTATE_LEFT
                        print("rotate_left")

                else:
                    # check for variety of angles
                    if self.r_middle - r_shoe > 30:
                        if self.state != MOVE_RIGHT:
                            self.state = MOVE_RIGHT
                            print("walk_right")
                        
                    elif self.r_middle - r_shoe < -30:
                        if self.state != MOVE_LEFT:
                            self.state = MOVE_LEFT
                            print("walk_left")

                    else:
                        if self.state != FORWARD:
                            self.state = FORWARD
                            print("FORWARD")

                # cv2.circle(parallel_image, (x_mid_shoe, y_mid_shoe), 10, (0, 0, 255), -1)
                # cv2.circle(parallel_image, (mid_shoe, local_max_y), 10, (0, 0, 255), -1)

    
                # cv2.circle(parallel_image, (int(self.x_middle), int(self.y_middle+150)), 10, (0, 255, 255), -1)
                # cv2.circle(parallel_image, (self.x_middle_next, self.y_middle_next), 10, (0, 255, 255), -1)
        

            # Display the result images
            cv2.imshow("Result Image cropped", result_image[self.gap_1:self.gap_2, :, :])
            cv2.imshow("path", parallel_image)

            if cv2.waitKey(33) == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()

def main():
    image_processor = ImageProcessor()
    image_processor.process_video("/home/redha/Cybathlon-Vision-Assistance-Race/output0.mp4")

if __name__ == '__main__':
    main()
