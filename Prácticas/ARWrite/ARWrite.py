import os
import string
import sys
import traceback
from collections import deque
import tkinter.simpledialog as tk
import tkinter

import cv2
import numpy as np
from keras.models import load_model

import parameters


def get_user_input(text):
    # Create root window and hide it, as we are not using it.
    root = tkinter.Tk()
    root.withdraw()
    result = "*"
    while len(result) != 1 or result not in string.ascii_lowercase or result == "":
        # Ask the user
        result = tk.askstring("ARWrite", text)
        if result:
            result = result.lower()
        else:
            return None
    return result


def get_target_letter(text):
    letter = get_user_input(text)
    if not letter:
        exit(0)
    # Take the image
    result = cv2.imread("{}{}.png".format(parameters.LETTERS_PATH, letter))
    return cv2.resize(result, parameters.WINDOW_SIZE), letter


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
neural_network = None

RETRY_ON_EXCEPTION = 1
PRINT_EXCEPTIONS = 0
SHOW_BLACKBOARD = 0
SHOW_GRAYSCALE_BLACKBOARD = 0

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

# Define a deque to store what is drawn
points = deque(maxlen=512)

# Variable for predictions.
prediction = None

# Start the camera
camera = cv2.VideoCapture(0)

close = False

ask_text = "Welcome to ARWrite! Please write a target letter"

while not close:
    """
    Step 1: ask the user to choose a letter
    """
    img, target_letter = get_target_letter(ask_text)
    """
    Step 2: initialize AR stuff
    """
    correct = False
    while not correct:
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
                # Draw a line to connect the points on the frame and the blackboard.
                for i in range(1, len(points)):
                    current_point = points[i]
                    previous_point = points[i - 1]
                    cv2.line(frame, previous_point, current_point, parameters.DRAWING_COLOR, parameters.DRAWING_THICKNESS)
                    cv2.line(blackboard, previous_point, current_point, parameters.WHITE_COLOR,
                             parameters.BACKGROUND_THICKNESS)
            else:
                # If there are not matches, the marker is not present in the image anymore, so it's time to analyse
                # the drawing (if we have points)
                if points:
                    # Create a greyscale blackboard to generate a binary image
                    grayscale_blackboard = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                    # Use Outsu's binarization (https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html):
                    blur = cv2.GaussianBlur(grayscale_blackboard, parameters.KERNEL_SIZE, 0)
                    threshold = cv2.threshold(grayscale_blackboard, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    # Find the points
                    grayscale_blackboard_points = cv2.findContours(threshold.copy(), cv2.RETR_TREE,
                                                                   cv2.CHAIN_APPROX_NONE)[0]
                    # If there are points in the greyscale blackboard, process them
                    if grayscale_blackboard_points:
                        # Find the marker
                        marker = max(grayscale_blackboard_points, key=cv2.contourArea)
                        # Resize the drawing area to the letter size leaving a threshold of 10 px
                        x, y, width, height = cv2.boundingRect(marker)
                        grayscale_blackboard = grayscale_blackboard[y - 10:y + height + 10, x - 10:x + width + 10]
                        # Resize the image to the origin one
                        grayscale_blackboard = cv2.resize(grayscale_blackboard,
                                                          (parameters.IMG_WIDTH, parameters.IMG_HEIGHT))
                        grayscale_blackboard = np.array(grayscale_blackboard)
                        # Make it binary
                        grayscale_blackboard = grayscale_blackboard.astype('float32') / 255
                        if SHOW_GRAYSCALE_BLACKBOARD:
                            cv2.imshow("Grayscale blackboard", grayscale_blackboard)
                        # Send it to the neural network
                        prediction = neural_network.predict(grayscale_blackboard.reshape(1, 28, 28, 1))[0]
                        prediction = np.argmax(prediction)
                    # Clear points and blackboard
                    points.clear()
                    blackboard = np.zeros(parameters.BLACKBOARD_SIZE, dtype=np.uint8)

            # Rectangle for prediction status background
            cv2.rectangle(frame, parameters.RECTANGLE_P1, parameters.RECTANGLE_P2, color=parameters.RECTANGLE_COLOR,
                          thickness=cv2.FILLED)
            # Show the prediction on screen
            if prediction:
                predicted_letter = string.ascii_lowercase[prediction]
                if predicted_letter == target_letter:
                    correct = True
                    ask_text = "Enter another target letter if you want to play again!"
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
            concatenation = np.concatenate((frame, img), axis=1)
            cv2.imshow("ARWrite", concatenation)
            if SHOW_BLACKBOARD:
                cv2.imshow("Blackboard", blackboard)
            # 'q' button will close the app
            if cv2.waitKey(1) & 0xFF == ord("q"):
                close = True
                correct = True
        except Exception as e:
            if PRINT_EXCEPTIONS:
                print("Exception on line {}\n".format(sys.exc_info()[2].tb_lineno))
                print(traceback.format_exc())
            if RETRY_ON_EXCEPTION:
                continue
            else:
                close = True

camera.release()
cv2.destroyAllWindows()
