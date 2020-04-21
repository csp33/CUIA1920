# Neural network training
DATA_PATH = 'data/'
NEURAL_NETWORK_FILE = 'neural_network/neural_network.h5'
BATCH_SIZE = 128
NUMBER_OF_EPOCHS = 10

# Dataset parameters (https://www.nist.gov/itl/products-and-services/emnist-dataset)
NUMBER_OF_LETTERS = 26
IMG_WIDTH = 28
IMG_HEIGHT = 28

# Marker parameters (BLUE)
LOWER_BOUND = [100, 60, 60]
UPPER_BOUND = [140, 255, 255]

# Sizes
KERNEL_SIZE = (5, 5)
BLACKBOARD_SIZE = (480, 640, 3)
DRAWING_AREA_SIZE = (200, 200, 3)
WINDOW_SIZE = (500, 500)

# Colors
DRAWING_COLOR = (0, 255, 255)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
OK_COLOR = (0, 255, 0)
WRONG_COLOR = (0, 0, 255)
RECTANGLE_COLOR = WHITE_COLOR

# Thicknesses
BACKGROUND_THICKNESS = 8
DRAWING_THICKNESS = 2

# Positions
TEXT_POSITION = (30, 470)
RECTANGLE_P1 = (0, 440)
RECTANGLE_P2 = (499, 490)
