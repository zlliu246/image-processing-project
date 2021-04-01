from proximity.utils.utils import predict, preprocess, postprocess, get_distance, draw_boxes, get_iou
import numpy as np
from keras.models import load_model
from proximity.models.layers import BilinearUpSampling2D


class ProximityModel:
    def __init__(self):
        self.owners = {}  # Keep track of the owners e.g. {"name": abc, "frames": 1, "obj_depth": 31, "closest_person": bca, "theft_counter":10}
        self.min_frames = 5  # minimum number of frames before determining owner
        self.faces = []  # list of multiple bboxes of faces/persons identified
        self.obj = []  # bbox of objs
        self.img = None  # curr image frame
        self.depth_thres = 30  # threshold of depth value for comparing 2 object's depth
        self.min_distance = 80  # minimum distance (pixels) between obj and person to consider it in proximity
        self.model = self.load_model(path='./proximity/models/nyu.h5')
        self.theft_counter_limit = 15  # If theft frames exceed 15, determine theft occurrence
        self.non_theft_counter_limit = 10
        self.iou_limit = 0.3  # If more than 30% of 2 bboxes overlap, then take only 1

    def detect_theft(self):
        """
        Checks if object has owner. If none, assign an owner. Else, check if nearest person is the owner
        Returns True if theft has been detected, False otherwise
        """
        for ind, obj in enumerate(self.obj):
            if not self.__has_owner(ind):
                self.__detect_owner(ind, obj)
            else:
                nearest = self.__get_nearest_person(ind, obj)
                if nearest is not None and self.owners[ind]['name'] != nearest:
                    self.owners[ind]['theft_counter'] += 1
                else:
                    self.owners[ind]['non_theft_counter'] += 1

                # If no theft for a period of time, then reset theft counter
                if self.owners[ind]['non_theft_counter'] > self.non_theft_counter_limit:
                    self.owners[ind]['theft_counter'] = 0
                    self.owners[ind]['non_theft_counter'] = 0

    def __get_nearest_person(self, ind, obj) -> str:
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
        (xmin, ymin), (xmax, ymax) = obj
        obj_depth = np.median(depth_map[ymin:ymax, xmin:xmax])
        for person in self.faces:
            (xmin, ymin), (xmax, ymax) = person[:2]
            cropped_depth = np.median(depth_map[ymin:ymax, xmin:xmax])
            # print(f"cropped depth [{ind}]:", cropped_depth)
            if abs(cropped_depth - obj_depth) <= self.depth_thres:
                candidate_lst.append(person + [cropped_depth])

        closest_person = self.get_closest_bbox(candidate_lst, obj)
        if self.owners.get(ind, None) is not None:  # Dont have to store closest person details if no owners established
            self.owners[ind]['closest_person'] = closest_person
            self.owners[ind]['obj_depth'] = obj_depth

        if closest_person:
            return closest_person[2]
        return closest_person

    def __detect_owner(self, ind, obj):
        nearest = self.__get_nearest_person(ind, obj)
        if self.owners.get(ind, None) is None:  # Initialize new owner with nearest person
            self.owners[ind] = {"name": nearest,
                                "frames": 1,
                                "obj_depth": None,
                                "closest_person": None,
                                "theft_counter": 0,
                                "non_theft_counter": 0}
        elif self.owners[ind]['name'] == nearest:  # If concurrent frames pointing to the same person
            self.owners[ind]['frames'] += 1
        else:  # Reset if interruption
            self.owners[ind] = None

    def update_curr(self, faces: list, obj: list, img: np.ndarray):
        """
        faces: list of body bounding boxes and face names e.g. [[(x1, y1), (x2, y2), "ZuoLin"],.....]
        obj: list of objects' coordinates of 1 bounding box e.g [[(x1, y1), (x2, y2)], ...]
        img: np array of current frame
        """
        self.faces = faces
        self.obj = self.preprocess_boxes(obj)
        self.img = img

    def preprocess_boxes(self, obj):
        initial_len = 0
        while initial_len != len(obj):
            initial_len = len(obj)
            for ind, o in enumerate(obj[:-1]):
                if get_iou(o, obj[ind+1]) > self.iou_limit:
                    obj.pop(ind+1)
                    break

        obj = sorted(obj, key=lambda tup: tup[0][0])
        print("obj:", obj)
        return obj

    def load_model(self, path):
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
        model = load_model(path, custom_objects=custom_objects, compile=False)
        return model

    def __has_owner(self, ind) -> bool:
        """
        Checks for presence of an owner (ensuring min consecutive frames met)
        """
        if self.owners.get(ind, None) is not None and self.owners[ind]['frames'] >= self.min_frames:
            return True
        return False

    def get_closest_bbox(self, candidates: list, obj: list):
        min_dist = float("inf")
        min_bbox = None
        for candidate in candidates:
            curr_dist = get_distance((candidate[0], candidate[1]), obj)
            # print("curr dist:", curr_dist)
            if curr_dist < min_dist and curr_dist < self.min_distance:
                min_dist = curr_dist
                min_bbox = candidate
        if min_bbox:
            return min_bbox + [min_dist]
        return None

    def draw_owner_name(self, img, ind, obj) -> np.ndarray:
        new_image = img.copy()
        return draw_boxes(new_image, *obj, "Owner:", self.owners[ind]['name'], 33)

    def draw_depth_proximity(self, img, ind, obj) -> np.ndarray:
        new_image = img.copy()
        # Draw depth
        new_image = draw_boxes(new_image, *obj, "Depth:", str(self.owners[ind]['obj_depth']), 53)
        new_image = draw_boxes(new_image, *self.owners[ind]['closest_person'][:2],
                               "Depth:", str(self.owners[ind]['closest_person'][3]),
                               -130)
        # Draw proximity (distance between the bboxes)
        new_image = draw_boxes(new_image, *self.owners[ind]['closest_person'][:2],
                               "bbox distance:", str(self.owners[ind]['closest_person'][4]),
                               -150)
        return new_image

    def draw_labels(self, img) -> np.ndarray:
        new_image = img.copy()
        for ind, obj in enumerate(self.obj):
            has_owner = self.__has_owner(ind)
            if has_owner:
                new_image = self.draw_owner_name(new_image, ind, obj)
                if self.owners[ind]['obj_depth'] is not None and self.owners[ind]['closest_person'] is not None:
                    new_image = self.draw_depth_proximity(new_image, ind, obj)
                if self.owners[ind]['theft_counter'] > self.theft_counter_limit:
                    new_image = self.draw_theft_alert(new_image)
        return new_image

    @staticmethod
    def draw_theft_alert(img) -> np.ndarray:
        new_image = img.copy()
        return draw_boxes(new_image, (0, 0), (0, 0), "Theft alert!", "", -50)
