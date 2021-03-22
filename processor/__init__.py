from processor.deepface_processor import *
from processor.yolo_processor import detect_objects_yolo

def process(img):
    """
    processes 1 image at a time, returns new image
    """

    # bboxs = [[20, 20, 20, 20, 0.99, "Laptop"]]
    # new_img = drawBBoxs(img, bboxs, col_dict)
    new_img = img

    try:new_img = detect_objects_yolo(new_img)
    except:print("yolo ERROR")

    try:new_img = detect_deepface(new_img)
    except:print("deepface ERROR")

    return new_img

