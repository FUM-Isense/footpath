import cv2
import numpy as np
import math
import pyttsx3
from queue import PriorityQueue
from skspatial.objects import Line, Points, Vector, Point

FORWARD = 0
ROTATE_RIGHT = 1
ROTATE_LEFT = 2
MOVE_RIGHT = 3
MOVE_LEFT = 4


class ImageProcessor:
    def __init__(self):
        self.x_middle, self.x_middle_next, self.y_middle, self.y_middle_next = (
            0,
            0,
            0,
            0,
        )
        self.gap_1 = 0
        self.gap_2 = 480
        self.state = FORWARD
        self.frame_counter = 0
        self.r_middle, self.theta_middle = 0, 0
        self.local_shoe_update = False
        self.global_theta, self.global_rho = 0, 360
        self.global_feedback_theta, self.global_feedback_rho = 0, 360
        self.new_path = True
        self.potential_path_counter = 0
        self.global_p1, self.global_p2 = (360, 150), (360, 250)
        self.global_feedback_p1, self.global_feedback_p2 = (360, 150), (360, 250)

    def draw_line_on_frame(self, rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 640 * (-b))
        y1 = int(y0 + 640 * (a))
        x2 = int(x0 - 640 * (-b))
        y2 = int(y0 - 640 * (a))

        return x1, y1, x2, y2

    def polar_to_cartesian(rho, theta):

        x = rho * math.cos(theta)
        y = rho * math.sin(theta)
        return x, y

    def cartesian_to_polar(self, x, y):
        rho = math.sqrt(x**2 + y**2)
        theta = math.atan2(y, x)
        return rho, theta

    def line_to_polar(self, x1, y1, x2, y2):
        # Calculate the slope (m) and y-intercept (b) of the line
        dx = x2 - x1
        dy = y2 - y1
        m = dy / dx
        b = y1 - m * x1

        # Calculate theta (angle in radians)
        theta = np.arctan(m)

        # Calculate rho (distance from origin to the line)
        rho = np.abs(b) / np.sqrt(1 + m**2)
        return rho, theta

    def contour_detection(self, frame, blurred):
        # Thresholding
        thresholded = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Morphological Operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(thresholded, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        # Contour Detection
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Shoe Detection
        local_min_x = float("inf")
        local_min_y = float("inf")
        local_max_x = float("-inf")
        local_max_y = float("-inf")
        self.local_shoe_updated = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Adjust the area threshold as needed
                # Check color similarity (assuming black shoes)
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y : y + h, x : x + w]
                avg_color = np.average(roi, axis=(0, 1))
                if (
                    avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50
                ):  # Adjust color threshold as needed
                    # Update the minimum and maximum coordinates
                    local_min_x = min(local_min_x, x)
                    local_min_y = min(local_min_y, y)
                    local_max_x = max(local_max_x, x + w)
                    local_max_y = max(local_max_y, y + h)
                    self.gap_1 = max(local_min_y - 290, 0)
                    self.gap_2 = max(local_min_y - 40, 0)
                    self.local_shoe_updated = True

                    if local_min_y == float("inf"):
                        self.gap_1 = 50
                        self.gap_2 = 300

        if self.local_shoe_updated:
            self.x_shoe = (local_min_x + local_max_x) // 2
            self.y_shoe = (local_min_y + local_max_y) // 2

    def intersect_line_with_forward_direction(self, rho1, theta1):

        # Create the coefficient matrix (A) and the constant vector (B)
        A = np.array([[np.cos(theta1), np.sin(theta1)], [1, 0]])  # forward line
        B = np.array([rho1, 360])  # or r_shoe

        # Solve the system of linear equations
        if np.linalg.det(A) != 0:
            intersection = np.linalg.solve(A, B)
            x, y = intersection[0], intersection[1]
            return x, y
        else:
            return None  # Lines are parallel or coincident

    def feedback(self, intersect_x, intersect_y):
        shoe_x = 250
        shoe_y = 360
        
        distance = math.sqrt((shoe_x - intersect_x) ** 2 + (shoe_y - intersect_y) ** 2)
        
        if distance < 150:
            print(1)
        elif distance > 150:
            print("Forward")
        
        
            # convert point to polar coordinate for comparison
            # r_shoe = (local_max_x + local_min_x) // 2
            # r_shoe = 360

            # audio_feedback = False
            # # check for off-center
            # if self.theta_middle > 0.2:
            #     if self.state != ROTATE_RIGHT:
            #         self.state = ROTATE_RIGHT
            #         print("rotate_right")

            #         if audio_feedback:
            #             pyttsx3.speak("right")

            # elif self.theta_middle < -0.2:
            #     if self.state != ROTATE_LEFT:
            #         self.state = ROTATE_LEFT
            #         print("rotate_left")

            #         if audio_feedback:
            #             pyttsx3.speak("left")

            # else:
            #     # check for variety of angles
            #     if self.r_middle - r_shoe > 30:
            #         if self.state != MOVE_RIGHT:
            #             self.state = MOVE_RIGHT
            #             print("walk_right")

            #             if audio_feedback:
            #                 pyttsx3.speak("walk right")

            #     elif self.r_middle - r_shoe < -30:
            #         if self.state != MOVE_LEFT:
            #             self.state = MOVE_LEFT
            #             print("walk_left")

            #             if audio_feedback:
            #                 pyttsx3.speak("walk left")

            #     else:
            #         if self.state != FORWARD:
            #             self.state = FORWARD
            #             print("FORWARD")
        return None

    def distance_to_line(self, x, y, rho, theta):
        return abs(rho - (x * math.cos(theta) + y * math.sin(theta)))

    def distance_point_to_parallel_lines(self, x, y, rho1, theta1, rho2, theta2):
        # Calculate the distance
        return abs(
            (rho1 - (x * math.cos(theta1) + y * math.sin(theta1)))
            + (rho2 - (x * math.cos(theta2) + y * math.sin(theta2)))
        )

    def direction_vector_to_polar(self, point, vector):
        point = Point([344.70588235, 99.11764706])
        direction = Vector([0.00827273, -0.99996578])

        # Calculate theta (the angle of the direction vector with respect to the x-axis)
        theta = np.arctan2(direction[1], direction[0])

        # Calculate the unit normal vector to the direction vector
        # The normal vector is (-direction[1], direction[0])
        normal_vector = Vector([-direction[1], direction[0]])
        normal_unit_vector = normal_vector / normal_vector.norm()

        # Calculate rho (the perpendicular distance from the origin to the line)
        rho = point.dot(normal_unit_vector)

        return rho, theta

    def draw_line(self, rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Calculate the start and end points of the line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        return x1, y1, x2, y2

    def average_line(self, lines):
        points = np.array([line.point for line in lines])
        directions = np.array([line.direction for line in lines])

        average_point = np.mean(points, axis=0)
        average_direction = np.mean(directions, axis=0)

        # Normalize the average direction
        average_direction = average_direction / np.linalg.norm(average_direction)

        return Line(Point(average_point), Vector(average_direction))

    def process_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        segment_size = (15, 10)
        point_threshold = 15

        # avg_points = []
        # avg_theta = []
        # avg_rho = []
        # for i in range(30):
        #     avg_theta.append(1.57)
        #     avg_rho.append(360)
        # # avg_theta[9] = -1
        # # avg_theta[-1] = -1
        # # avg_theta.pop(-1)
        # # print(avg_theta)
        # # avg_theta

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # # contour detection
            # self.contour_detection(frame, blurred)

            blur_image = blurred[self.gap_1 : self.gap_2]

            # Apply Canny edge detection
            edges = cv2.Canny(blur_image, 70, 120)

            # Apply Hough Transform for line detection
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)

            # Draw the detected lines on the original image
            result_image = frame.copy()
            result_image = result_image[self.gap_1 : self.gap_2]

            parallel_lines = []
            if lines is not None:
                for idx_1 in range(lines.shape[0]):
                    rho_1, theta_1 = lines[idx_1, 0]

                    for idx_2 in range(idx_1 + 1, lines.shape[0]):
                        rho_2, theta_2 = lines[idx_2, 0]

                        if abs(theta_1 - theta_2) < 0.2:
                            if abs(rho_1 - rho_2) > 100:
                                similar_lines_detected = False
                                for prev_line in parallel_lines:
                                    if (
                                        abs(prev_line[1] - theta_1) < 0.2
                                        and (abs(prev_line[0] - rho_1) < 20)
                                    ) or (
                                        abs(prev_line[1] - theta_2) < 0.2
                                        and (abs(prev_line[0] - rho_2) < 20)
                                    ):
                                        similar_lines_detected = True
                                        break

                                if not similar_lines_detected:
                                    parallel_lines.append((rho_1, theta_1))
                                    parallel_lines.append((rho_2, theta_2))
            # # Draw parallel lines
            # for line in parallel_lines:
            #     rho, theta = line
            #     x1, y1, x2, y2 = ImageProcessor.draw_line_on_frame(rho, theta)
            #     cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            gap_diff = self.gap_2 - self.gap_1
            point_to_line_distance = PriorityQueue()
            path_points = []
            # for i in range(0, gap_diff, segment_size[0]):
            #     for j in range(0, 640, segment_size[1]):
            for i in range(gap_diff, 0, -segment_size[0]):
                for j in range(639, 0, -segment_size[1]):
                    if blur_image[gap_diff - i - 1][j] < 150:
                        continue
                    for index_line in range(0, len(parallel_lines)):
                        # Using priority queue to get the least distance
                        point_to_line_distance.put(
                            (
                                self.distance_to_line(
                                    j,  # width
                                    gap_diff - i,  # height
                                    parallel_lines[index_line][0],  # rho
                                    parallel_lines[index_line][1],  # theta
                                ),
                                index_line,
                            )
                        )

                    # print(point_to_line_distance.qsize())
                    if not point_to_line_distance.empty():
                        index_parallel = point_to_line_distance.get(0)[1]
                        if index_parallel % 2 != 0:
                            index_parallel -= 1

                        distance_to_prallel_lines = (
                            self.distance_point_to_parallel_lines(
                                j,  # width
                                gap_diff - i,  # height
                                parallel_lines[index_parallel][0],  # rho1
                                parallel_lines[index_parallel][1],  # theta1
                                parallel_lines[index_parallel + 1][0],  # rho2
                                parallel_lines[index_parallel + 1][1],  # theta2
                            )
                        )
                        if distance_to_prallel_lines < 20:
                            #Draw points
                            # cv2.circle(
                            #     result_image,
                            #     (j, gap_diff - i),
                            #     5,
                            #     (255, 0, 255),
                            #     -1,
                            # )
                            path_points.append((j, gap_diff - i))
                    if len(path_points) == point_threshold:
                        break
                if len(path_points) == point_threshold:
                    break

            if len(path_points) == point_threshold:
                points = Points(path_points)
                line_fit = Line.best_fit(points)
                if line_fit.direction[0] == -0.0:
                    line_fit.direction[0] = -0.01
                    line_fit.direction[1] = -1.01

                p1 = np.array(line_fit.to_point(t=-150))
                p2 = np.array(line_fit.to_point(t=150))
                min_p = p1
                if p2[1] < p1[1]:
                    p1 = p2
                    p2 = min_p
                # p1 up, p2 down
                rho, theta = self.line_to_polar(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1])
                # adapt rho and theta to the frame
                theta -= np.pi / 2
                if theta < -np.pi / 2 and theta > -np.pi:
                    rho = -rho

                # try:
                #     x1, y1, x2, y2 = self.draw_line_on_frame(rho, theta)
                #     cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #     print(rho, theta)
                # except:
                #     print(line_fit)
                #     print("no fit_line")

                # checks for the staight lines
                if self.new_path or abs(abs(theta) - abs(self.global_theta)) < 0.2:
                    try:
                        if p1[1] < 300 and p2[1] < 300:
                            self.global_p1, self.global_p2 = p1, p2
                            self.global_theta = theta
                            self.global_rho = rho
                            intersect_x, intersect_y = 360, 150
                            # find the intersect point or forward line
                            if abs(self.global_theta) < 0.2:
                                intersect_x, intersect_y = 360, 150
                            else:
                                intersect_x, intersect_y = (
                                    self.intersect_line_with_forward_direction(
                                        self.global_rho, self.global_theta
                                    )
                                )
                                if (
                                    blur_image[intersect_x][intersect_y] < 150 # check for the point to be on the path
                                ) or intersect_y > 300: # check for false intersections
                                    intersect_x, intersect_y = 360, 150
                        
                            self.new_path = False
                    except:
                        print("out of image")

                # checks for potential new path
                else:
                    self.potential_path_counter += 1
                    if self.potential_path_counter == 15:
                        # todo , if you want add a mod calculation for best orientation
                        self.new_path = True
                        self.potential_path_counter = 0

                # cv2.line(result_image, (int(self.global_p1[0]), int(self.global_p1[1])), (int(self.global_p2[0]), int(self.global_p2[1])), (255, 0, 0), 5)

                # cv2.circle(result_image, (int(intersect_x), int(intersect_y)), 10, (0, 0, 255), -1)
                
                # Feedback
                shoe_x = 250
                shoe_y = 360
                             
                distance = math.sqrt((shoe_x - intersect_x) ** 2 + (shoe_y - intersect_y) ** 2)
                
                if distance < 250:
                    cv2.line(result_image, (int(self.global_p1[0]), int(self.global_p1[1])), (int(self.global_p2[0]), int(self.global_p2[1])), (0, 255, 0), 3)
                else:
                    cv2.line(result_image, (380, 150), (380, 350), (0, 255, 0), 3)
            

            cv2.imshow("path", result_image)

            if cv2.waitKey(10) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    image_processor = ImageProcessor()
    image_processor.process_video(
        "/home/redha/Projects/Vision_Assistance_Footpath/recordings/output0.mp4"
    )


if __name__ == "__main__":
    main()
