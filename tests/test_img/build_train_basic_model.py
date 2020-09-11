import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import the necessary libraries
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
import tensorflow as tf
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import Constant
# Fix random number for initial weight generation. This is important for reproducible.
import numpy as np
import random as rn

np.random.seed(0)
rn.seed(0)
import shutil

