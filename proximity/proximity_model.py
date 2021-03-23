from proximity.utils.utils import predict, preprocess, postprocess, get_distance, draw_boxes
import numpy as np
from keras.models import load_model
from proximity.models.layers import BilinearUpSampling2D


class ProximityModel:
    def __init__(self):
        self.owner = None  # Keep track of the owner
        self.min_frames = 15  # minimum number of frames before determining owner
        self.frames = 0  # curr frame count
        self.faces = []  # list of multiple bboxes of faces/persons identified
        self.obj = []  # bbox of obj (only 1 object is supported)
        self.img = None  # curr image frame
        self.depth_thres = 15  # threshold of depth value for comparing 2 object's depth
        self.closest_person = None
        self.obj_depth = None
        self.min_distance = 80  # minimum distance (pixels) between obj and person to consider it in proximity
        self.model = self.load_model(path='./proximity/models/nyu.h5')

    def detect_theft(self) -> bool:
        """
        Checks if object has owner. If none, assign an owner. Else, check if nearest person is the owner
        Returns True if theft has been detected, False otherwise
        """
        if not self.__has_owner():
            self.__detect_owner()
        else:
            nearest = self.__get_nearest_person()
            if nearest and self.owner != nearest:
                return True
        return False

    def __get_nearest_person(self) -> str:
        """
        Converts current frame into depth map (lower values indicate closer proximity)
        Crops out persons and get the median depth value and compares to object depth value.
        Select only those with similar depth and bounding box closest to object
        returns: name of nearest person to given object
        """
        inputs = preprocess(self.img)
        depth_map = predict(self.model, inputs)
        depth_map = postprocess(depth_map, self.img.shape[1], self.img.shape[0])

        candidate_lst = []
        (xmin, ymin), (xmax, ymax) = self.obj
        obj_depth = np.median(depth_map[ymin:ymax, xmin:xmax])
        for person in self.faces:
            (xmin, ymin), (xmax, ymax) = person[:2]
            cropped_depth = np.median(depth_map[ymin:ymax, xmin:xmax])
            if abs(cropped_depth-obj_depth) < self.depth_thres:
                candidate_lst.append(person+[cropped_depth])

        closest_person = self.get_closest_bbox(candidate_lst, self.obj)
        self.closest_person = closest_person
        self.obj_depth = obj_depth
        if closest_person:
            return closest_person[2]
        return closest_person

    def __detect_owner(self):
        nearest = self.__get_nearest_person()
        if not self.owner:  # Initialize new owner with nearest person
            self.owner = nearest
        elif self.owner == nearest:  # If concurrent frames pointing to the same person
            self.frames += 1
        else:  # Reset if interruption
            self.frames = 0

    def update_curr(self, faces: list, obj: list, img: np.ndarray):
        """
        faces: list of body bounding boxes and face names e.g. [[(x1, y1), (x2, y2), "ZuoLin"],.....]
        obj: coordinates of 1 bounding box e.g [(x1, y1), (x2, y2)]
        img: np array of current frame
        """
        self.faces = faces
        self.obj = obj
        self.img = img

    def load_model(self, path):
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
        model = load_model(path, custom_objects=custom_objects, compile=False)
        return model

    def __has_owner(self) -> bool:
        """
        Checks for presence of an owner (ensuring min consecutive frames met)
        """
        if self.owner and self.frames >= self.min_frames:
            return True
        return False

    def get_closest_bbox(self, candidates: list , obj: list):
        min_dist = float("inf")
        min_bbox = None
        for candidate in candidates:
            curr_dist = get_distance((candidate[0], candidate[1]), obj)
            if curr_dist < min_dist and curr_dist < self.min_distance:
                min_dist = curr_dist
                min_bbox = candidate
        if min_bbox:
            return min_bbox + [min_dist]
        return None

    def draw_owner_name(self, img) -> np.ndarray:
        return draw_boxes(img, *self.obj, "Owner:", self.owner, 33)

    def draw_depth_proximity(self, img) -> np.ndarray:
        new_image = img.copy()
        # Draw depth
        new_image = draw_boxes(new_image, *self.obj, "Depth:", str(self.obj_depth), 53)
        new_image = draw_boxes(new_image, *self.closest_person[:2], "Depth:", str(self.closest_person[3]), 53)
        # Draw proximity (distance between the bboxes)
        new_image = draw_boxes(new_image, *self.closest_person[:2], "bbox distance:", str(self.closest_person[4]), -33)
        return new_image

    def draw_labels(self, img) -> np.ndarray:
        new_image = img.copy()
        if self.__has_owner:
            new_image = self.draw_owner_name(new_image)
        if self.obj_depth and self.closest_person:
            new_image = self.draw_depth_proximity(new_image)
        return new_image

    def draw_theft_alert(self, img) -> np.ndarray:
        new_image = img.copy()
        return draw_boxes(new_image, (0, 0), (0, 0), "Theft alert!", "", -50)