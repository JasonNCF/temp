import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import json

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        return data
    

def get_metadata(d):
    path = d + '/_annotations.coco.json'
    data = read_json(path)
    imgs = data["images"]
    annos = data["annotations"]
    
    metadata = []
    for dic in imgs:
        temp_dict = dict()
        temp_dict["file_name"] = d + '/' + dic["file_name"]
        temp_dict["height"] = dic["height"]
        temp_dict["width"] = dic["width"]
        temp_dict["image_id"] = dic["id"]
        
        annotations = []
        meet = False
        for dic2 in annos:
            if dic2["image_id"] == dic["id"]:
                meet = True
                annotation = dict()
                annotation["bbox"] = dic2["bbox"] 
                annotation["bbox_mode"] = BoxMode.XYWH_ABS 
                annotation["category_id"] = dic2["category_id"] 
                annotation["segmentation"] = dic2["segmentation"]
                annotation["iscrowd"] = dic2["iscrowd"]
                annotations.append(annotation)
            elif dic2["image_id"] != dic["id"]:
                if meet:
                    break
                else:
                    continue
        temp_dict["annotations"] = annotations
        metadata.append(temp_dict)
    
    return metadata


if __name__ == "__main__":
    for d in ["train", "val"]:
        DatasetCatalog.register("pascal_" + d, lambda d=d: get_metadata("pascal/" + d))
        MetadataCatalog.get("pascal_" + d).set(thing_classes=["VOC", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                                                            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                                                            "sheep", "sofa", "train", "tvmonitor"])

    pascal_metadata = MetadataCatalog.get("pascal_train")

    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("pascal_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("pascal_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "pascal_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))