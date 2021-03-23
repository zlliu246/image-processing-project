import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
from deepface.commons.functions import initialize_detector, load_image, detect_face2

from processor import process

# constants
VIDEO_PATH = "dataset/demo_10.mp4"
OUTPUT_PATH = "output.avi"
FPS = 30

current_frame = 1
cap = cv2.VideoCapture(VIDEO_PATH)

ret,frame = cap.read()
frame_width = frame.shape[1]
frame_height = frame.shape[0]

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc("M", "J", "P", "G"), FPS, (frame_width, frame_height))
out.write(frame)

# parsing video frame by frame
while cap.isOpened():
    current_frame += 1
    print(f"PROCESSING FRAME {current_frame}")
    
    ret, frame = cap.read()
    if ret:
        
        # IMAGE FRAME GOES HERE
        img = load_image(frame)

        """
        ATTENTION: All your model processing goes into the "process" function
        """
        new_img = process(img)

        out.write(new_img)

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\ndone\n")