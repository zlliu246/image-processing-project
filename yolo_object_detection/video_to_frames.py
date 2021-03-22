from yolo import *
import tensorflow as tf
import cv2

video_path = "../video.mp4"

vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()

count = 0
while success:
    cv2.imwrite(f"frames/frame{count}.jpg", image)
    success, image = vidcap.read()
    count += 1

    print(count, end="\r")

