if __name__ == "__main__":
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("pascal_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "pascal_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))