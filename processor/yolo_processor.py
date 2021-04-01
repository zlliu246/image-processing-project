from yolo_object_detection.yolo import *

h5_path = "yolo_object_detection/largestuff/yolov3.h5"
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

wanted_labels = "person laptop".split(" ")

yolov3 = make_yolov3_model()
yolov3.load_weights(h5_path)

def detect_objects_yolo(image):
    image_h, image_w, _ = image.shape
    new_image = preprocess_input(image, net_h, net_w)

    yolos = yolov3.predict(new_image)

    boxes = []

    for i in range(len(yolos)):
        # decode the output of the network
        boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)   

    # draw bounding boxes on the image using labels
    draw_boxes(image, boxes, labels, obj_thresh, wanted_labels)
    person_boxes, object_boxes = get_wanted_boxes(boxes, labels, obj_thresh, wanted_labels)
    person_boxes = resize_boxes(person_boxes, image_h, image_w, 0.3)

    return (image).astype('uint8'), person_boxes, object_boxes