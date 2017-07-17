# --------------------------------------------------------------------------------------------
# import statements
# --------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import re
import nltk
import sys
import time
import math
import random
import os

# --------------------------------------------------------------------------------------------
# initializing constants and flags
# --------------------------------------------------------------------------------------------
TEST_MODE = False
TRAIN_MODE = False
if sys.argv[1] == '--train':
  TRAIN_MODE = True
elif sys.argv[1] == '--test-interactive':
  TEST_MODE = True

MAX_INPUT_LENGTH = 10
BATCH_SIZE = 10
LEARNING_RATE = 0.001
SAVE_EVERY_N_STEP = 200
STEPS_PER_CKPT = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EMBEDDING_SIZE = 32
SAVE_FILE = 'gotmodel.save'
NUM_STEPS_FILE = '.numsteps'


