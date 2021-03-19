from yolo import *
import tensorflow as tf

weights_path = "yolov3.weights"
image_path = "images/zltest.jpg"

# set some parameters
net_h, net_w = 416, 416
obj_thresh, nms_thresh = 0.5, 0.45
anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# make the yolov3 model to predict 80 classes on COCO
yolov3 = tf.keras.models.load_model("yolov3.h5")

# preprocess the image
# image = cv2.imread(image_path)
# image_h, image_w, _ = image.shape
# new_image = preprocess_input(image, net_h, net_w)

# run the prediction
# yolos = yolov3.predict(new_image)

import cv2
vidcap = cv2.VideoCapture('videos/video.mp4')
success,image = vidcap.read()
count = 0
while success:
  	cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
	success, image = vidcap.read()
	print(image.shape)

