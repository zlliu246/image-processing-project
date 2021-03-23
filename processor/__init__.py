from processor.deepface_processor import *
from processor.yolo_processor import detect_objects_yolo
from proximity.proximity_model import ProximityModel

proximity_model = ProximityModel()


def process(img):
    """
    processes 1 image at a time, returns new image
    """

    # bboxs = [[20, 20, 20, 20, 0.99, "Laptop"]]
    # new_img = drawBBoxs(img, bboxs, col_dict)
    new_img = img
    person_face_detected = None

    try:
        new_img, person_boxes, object_boxes = detect_objects_yolo(new_img)
    except:
        print("yolo ERROR")

    try:
        # Modified to detect on cropped images and output [person bbox, face_name] to track corresponding face to body
        new_img, person_face_detected = detect_deepface_cropped(new_img, person_boxes)
    except:
        print("deepface ERROR")

    if person_face_detected and len(object_boxes) > 0:
        print("Running proximity model")
        # Only take first object bbox, assume only 1 object in frame
        proximity_model.update_curr(person_face_detected, object_boxes[0], img)

        if proximity_model.detect_theft():
            print("Theft detected")
            new_img = proximity_model.draw_theft_alert(new_img)
        new_img = proximity_model.draw_labels(new_img)

    return new_img
