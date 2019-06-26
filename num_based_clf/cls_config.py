from easydict import EasyDict as edict

cls_config = edict()
cls_config.check_model_path = "check.model"
cls_config.cls_model_path = "cls.model"
cls_config.feature_order = ['hsil', 'lsil','ec']
cls_config.cls_order = ['hsil', 'lsil']
cls_config.thresh = .1