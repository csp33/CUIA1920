import os
import string
import sys
import traceback
from collections import deque

import cv2
import numpy as np
from keras.models import load_model

import parameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
neural_network = None

RETRY_ON_EXCEPTION = 1
PRINT_EXCEPTIONS = 1

# Load the previously compiled model
try:
    neural_network = load_model(parameters.NEURAL_NETWORK_FILE)
except:
    print("Error reading model. You must compile and train it first.\n")
    exit(-1)

# Define lower and upper bounds to recognise a color
lower_bound = np.array(parameters.LOWER_BOUND)
upper_bound = np.array(parameters.UPPER_BOUND)

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones(parameters.KERNEL_SIZE, np.uint8)

# Define the blackboard
blackboard = np.zeros(parameters.BLACKBOARD_SIZE, dtype=np.uint8)
# Define the drawing area
drawing_area = np.zeros(parameters.DRAWING_AREA_SIZE, dtype=np.uint8)

# Define a deque to store what is drawn
points = deque(maxlen=512)

# Variable for predictions.
prediction = None

# Start the camera
camera = cv2.VideoCapture(0)

close = False

"""
Step 1: ask the user to choose a letter
"""
# TODO
img = cv2.imread('a.png')
img = cv2.resize(img, parameters.WINDOW_SIZE)
target_letter = 's'
"""
Step 2: initialize AR stuff
"""


while not close:
    try:
        (grabbed, frame) = camera.read()
        # Horizontally flip the frame and resize it
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, parameters.WINDOW_SIZE)
        # Convert the colors to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Apply erosion and dilation to find the marker
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        matches, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check if there are matches
        if matches:
            # The marker should be the biggest match, so we take it.
            marker = max(matches, key=cv2.contourArea)
            # Get the minimum enclosing circle around the marker and draw it
            (x, y), radius = cv2.minEnclosingCircle(marker)
            cv2.circle(frame, (int(x), int(y)), int(radius), parameters.DRAWING_COLOR, 2)
            # Calculate the center of the circle
            M = cv2.moments(marker)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            # Add it to our points deque
            points.appendleft(center)
        else:
            # If there are not matches, the marker is not present in the image anymore, so it's time to analyse
            # the drawing (if we have points)
            if points:
                # Draw the blackboard
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, parameters.KERNEL_SIZE, 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_points = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
                if blackboard_points:
                    marker = max(blackboard_points, key=cv2.contourArea)
                    if cv2.contourArea(marker) > 1000:
                        x, y, w, h = cv2.boundingRect(marker)
                        drawing_area = blackboard_gray[y - 10:y + h + 10, x - 10:x + w + 10]
                        image = cv2.resize(drawing_area, (parameters.IMG_WIDTH, parameters.IMG_HEIGHT))
                        image = np.array(image)
                        image = image.astype('float32') / 255
                        prediction = neural_network.predict(image.reshape(1, 28, 28, 1))[0]
                        prediction = np.argmax(prediction)
                # Clear points and blackboard
                points.clear()
                blackboard = np.zeros(parameters.BLACKBOARD_SIZE, dtype=np.uint8)

        # Draw a line to connect the points on the frame and the blackboard.
        for i in range(1, len(points)):
            current_point = points[i]
            previous_point = points[i - 1]
            cv2.line(frame, previous_point, current_point, parameters.DRAWING_COLOR, parameters.DRAWING_THICKNESS)
            cv2.line(blackboard, previous_point, current_point, parameters.WHITE_COLOR, parameters.BACKGROUND_THICKNESS)

        # Rectangle for prediction status background
        cv2.rectangle(frame, parameters.RECTANGLE_P1, parameters.RECTANGLE_P2, color=parameters.RECTANGLE_COLOR,
                      thickness=cv2.FILLED)
        # Show the prediction on screen
        if prediction:
            predicted_letter = string.ascii_lowercase[prediction]
            if predicted_letter == target_letter:
                text = "Good job! You successfully draw letter {}".format(predicted_letter)
                color = parameters.OK_COLOR
            else:
                text = "No! You draw {} but you had to draw {}".format(predicted_letter, target_letter)
                color = parameters.WRONG_COLOR
            cv2.putText(frame, text, parameters.TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:

            cv2.putText(frame, "Please write letter {} with a valid marker".format(target_letter),
                        parameters.TEXT_POSITION,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, parameters.BLACK_COLOR, 2)
        # Show the frame
        concatenation = np.concatenate((frame, img), axis=1)
        cv2.imshow("ARWrite", concatenation)

        # If 'q' key is pressed, close the app
        if cv2.waitKey(1) & 0xFF == ord("q"):
            close = True
    except Exception as e:
        if PRINT_EXCEPTIONS:
            print("Exception on line {}\n".format(sys.exc_info()[2].tb_lineno))
            print(traceback.format_exc())
        if RETRY_ON_EXCEPTION:
            continue
        else:
            exit(-1)

camera.release()
cv2.destroyAllWindows()
