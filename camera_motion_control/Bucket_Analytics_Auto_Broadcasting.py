"""
File: Bucket_Analytics_Auto_Broadcasting_Branch
Author: Yu-Ju Fang
## Bucket Analytics Auto Broadcasting Branch

### Main idea
  - Using yolo model to distinguish the postion of each players on the court and save the postion to a txt file and then
  Use the attached file to find the position that contains most players and then zoom in based on the postion we found.

### Architecture diagram
  - Step 1: Read the txt file that records the position of each plaer in each frame(read_txt function)
  - Step 2: Project all the players' position onto the x-axis and find the max_x_position using sliding window
  (find_max_x function)
  - Step 3: Zoom in the frame using max_x_position as core(zoom function)
"""
import cv2
from collections import defaultdict


def read_text(file_name):
    """
    :param file_name: Read the file we get from using yolo model to find each players' position in each frame
    :return: all the player positions in each frame
    """

    player_positions = defaultdict(list)
    with open(file_name) as f:
        contents = f.readlines()

    for i in range(len(contents)):
        lst = contents[i].split()
        player_positions[int(lst[0])].append(lst[2:6])

    return player_positions


def find_max_x(project_lst):
    """
    Find the position on x-axis that indicates most players are by using a sliding window with 300 pixel width
    :return : the position that we want to zoom in
    """
    sum_project_lst = [0] * WIDTH
    for i in range(200, len(project_lst) - 200):
        sum_project_lst[i] = sum(project_lst[i - 150: i + 150])

    max_val = max(sum_project_lst)
    return sum_project_lst.index(max_val)


def zoom(max_x_position, frame1, radius_y, radius_x, center_y):
    """
    Zoom in the frame
    :param max_x_position: the position we are going to zoom in
    :param frame1: the current frame
    :param radius_x: the width of the zoomed in frame
    :param radius_y: the height of the zoomed in frame
    :param center_y: the y-axis position of zoomed in frame
    :return: the Zoomed frame
    """
    center_x = int(max_x_position)

    min_y, max_y = center_y - radius_y, center_y + radius_y
    min_x, max_x = center_x - radius_x, center_x + radius_x

    cropped = frame1[min_y:max_y, min_x:max_x]
    resized_cropped = cv2.resize(cropped, (WIDTH, HEIGHT))

    return resized_cropped


cap = cv2.VideoCapture('test3.mp4')  # Read the video
frame_dict = read_text('test3.txt')  # Read the txt file which stores the position of each player in each frame
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Constants should be customized
UPPER_COURT_BOUNDARY = 300  # upper boundary of the basketball court
LOWER_COURT_BOUNDARY = 450  # lower boundary of the basketball court
LEFT_COURT_BOUNDARY = 550
RIGHT_COURT_BOUNDARY = 700
FIXED_POSITION_LEFT_COURT = 350
FIXED_POSITION_RIGHT_COURT = 930

# Constants better be fixed
BROADCAST_WIDTH = 320  # This should be smaller than LEFT_COURT_BOUNDARY and also smaller than WIDTH - RIGHT_COURT_BOUNDARY
BROADCAST_HEIGHT = 180  # This should be smaller than UPPER_COURT_BOUNDARY and also smaller than HEIGHT - LOWER_COURT_BOUNDARY
ZOOM_IN_BROADCAST_WIDTH = 250
ZOOM_IN_BROADCAST_HEIGHT = 150

ZOOM_IN_OUT_SPEED_WIDTH = 3  # Determine how fast we zoom in
ZOOM_IN_OUT_SPEED_HEIGHT = 2
TRANSITION_SPEED = 6
COURT_SPEED = 4
NORMAL_SPEED = 2

# The processed video will be save in current file due to the code below
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (WIDTH, HEIGHT))


def main():
    ret, frame1 = cap.read()
    project_lst = [0] * WIDTH
    first = True
    frame_number = 3

    # Determine the scale of zoom in
    zoom_width = BROADCAST_WIDTH
    zoom_height = BROADCAST_HEIGHT
    center_y = int(HEIGHT / 2)

    while cap.isOpened():

        # Use yolo to capture player position and project to the x-axis
        for people in frame_dict[frame_number]:
            for pos in range(int(people[0]), int(people[0]) + int(people[2])):

                # Only players on the court should be include to determine max_x_position
                if int(people[1]) + int(people[3]) / 2 > UPPER_COURT_BOUNDARY and \
                        int(people[1]) + int(people[3]) < LOWER_COURT_BOUNDARY:
                    project_lst[pos] += int(people[3])

        # Find the start point
        max_x_position = find_max_x(project_lst)

        if first:
            prev_max_position = max_x_position
            first = False

        if WIDTH - 100 > max_x_position > 100:  # Valid Max Position

            # Decide how fast the frame should move due to the max_x_position
            if FIXED_POSITION_RIGHT_COURT > max_x_position > FIXED_POSITION_LEFT_COURT:
                speed = TRANSITION_SPEED  # Speed is customized
            else:  # left court or right court speed
                speed = COURT_SPEED  # Speed is customized
            # Only move when max_x_position moves 100 pixel large than previous max position

            # Decide the position to zoom in based on the max_x_position - pre_max_position
            if max_x_position - prev_max_position > 40:
                max_x_position = prev_max_position + speed
                prev_max_position = max_x_position
            elif max_x_position - prev_max_position < -40:
                max_x_position = prev_max_position - speed
                prev_max_position = max_x_position
            else:
                if max_x_position - prev_max_position >= 0:
                    max_x_position = prev_max_position + NORMAL_SPEED
                else:
                    max_x_position = prev_max_position - NORMAL_SPEED
                prev_max_position = max_x_position

            # If max position is less than LEFT_COURT_BOUNDARY we fix the position at FIXED_POSITION_LEFT_COURT
            # which means all the players are at the left court
            if max_x_position < LEFT_COURT_BOUNDARY:
                if max_x_position > FIXED_POSITION_LEFT_COURT:
                    max_x_position -= 1
                    if zoom_height < BROADCAST_HEIGHT and zoom_width < BROADCAST_WIDTH:
                        zoom_width += ZOOM_IN_OUT_SPEED_WIDTH
                        zoom_height += ZOOM_IN_OUT_SPEED_HEIGHT
                    if center_y <= int(HEIGHT / 2):
                        center_y += 1
                else:
                    max_x_position = FIXED_POSITION_LEFT_COURT
                    # If max_position is on the FIXED_POSITION_LEFT_COURT then zoom in
                    if zoom_height > ZOOM_IN_BROADCAST_HEIGHT and zoom_width > ZOOM_IN_BROADCAST_WIDTH:
                        zoom_width -= ZOOM_IN_OUT_SPEED_WIDTH
                        zoom_height -= ZOOM_IN_OUT_SPEED_HEIGHT
                    if center_y >= int(HEIGHT / 2) - 5:
                        center_y -= 1

            # If max position is greater than RIGHT_COURT_BOUNDARY we fix the position at FIXED_POSITION_RIGHT_COURT
            # since all the players are at the right court
            if max_x_position > RIGHT_COURT_BOUNDARY:
                if max_x_position < FIXED_POSITION_RIGHT_COURT:
                    max_x_position += 1
                    if zoom_height < BROADCAST_HEIGHT and zoom_width < BROADCAST_WIDTH:
                        zoom_width += ZOOM_IN_OUT_SPEED_WIDTH
                        zoom_height += ZOOM_IN_OUT_SPEED_HEIGHT
                    if center_y <= int(HEIGHT / 2):
                        center_y += 1
                else:
                    max_x_position = FIXED_POSITION_RIGHT_COURT
                    # If max_position is on the FIXED_POSITION_RIGHT_COURT then zoom in
                    if zoom_height > ZOOM_IN_BROADCAST_HEIGHT and zoom_width > ZOOM_IN_BROADCAST_WIDTH:
                        zoom_width -= ZOOM_IN_OUT_SPEED_WIDTH
                        zoom_height -= ZOOM_IN_OUT_SPEED_HEIGHT
                    if center_y >= int(HEIGHT / 2) - 5:
                        center_y -= 1

            # Zoom in to the max_x_position
            frame1 = zoom(max_x_position, frame1, zoom_height, zoom_width, center_y)

        else:  # invalid Max Position
            frame1 = zoom(prev_max_position, frame1, zoom_height, zoom_width, center_y)

        project_lst = [0] * WIDTH  # Initialize project_lst
        cv2.imshow('Basketball Action Detection', frame1)
        if ret is True:
            out.write(frame1)

        ret, frame1 = cap.read()
        frame_number += 1
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()


if __name__ == '__main__':
    main()