from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

from cls_config import cls_config

def load_model(file_path):
    with open(file_path, 'rb') as f:
       clf = pickle.dump(f)

    return clf

def update_result(result, box_thresh):
    new_result = {}
    for k in result:
        label_boxes = result[k]
        label_boxes = np.array(label_boxes) if type(label_boxes) == type(list()) else label_boxes

        if label_boxes.shape[0] > 0:
            label_boxes = label_boxes[label_boxes[:, -1] >= box_thresh]
        new_result[k] = label_boxes
    return new_result

class img_cls_clf:
    def __init__(self):
        self.check_clf = load_model(cls_config.check_model_path)
        self.cls_clf = load_model(cls_config.cls_model_path)
        self.feature_order = cls_config.feature_order
        self.cls_order = cls_config.cls_order
        self.thresh = cls_config.thresh

    def infer(self, result):
        test_data = []
        x = []
        new_result = update_result(result)
        for s_feature in self.feature_order:
            x.append(len(new_result[s_feature]))
        test_data.append(x)
        test_data = np.array(test_data, dtype=np.float32)

        pos_check = self.check_clf.predict(test_data)[0]
        if pos_check == 0:
            return 'normal'

        cls_label_index = self.cls_clf.predict(test_data)[0]
        return self.cls_order[cls_label_index]
