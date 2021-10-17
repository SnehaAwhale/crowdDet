#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: sudhanshukumar
"""



# from torchvision.models import detection
# from detectron2.engine import DefaultPredictor
# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

# import some common libraries
# import numpy as np
# import os, json, cv2, random
# from google.colab.patches import cv2_imshow
#
# # import some common detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
# import megengine as mge
# from keras.models import load_model
# from keras.preprocessing import image

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        # model = load_model('model.h5')
        # model = load_model('best.pt')

        # summarize model
        # model.summary()
        # imagename = self.filename
        # test_image = image.load_img(imagename, target_size = (64, 64))
        # test_image = image.img_to_array(test_image)
        # test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        # model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Sneha/Downloads/Cat_Dog-main/best.pt',
        # force_reload = True)
        # model = torch.hub.load('MegEngine/Models', 'faster_rcnn_res50_coco_3x_800size', pretrained=True).eval()
        # model = torch.utils.model_zoo.load_url('configs/COCO-Detection', 'faster_rcnn_R_50_FPN_3x', path='C:/Users/Sneha/Downloads/Cat_Dog-main/Detectron/model_final (2).pth',
        #                        force_reload=True)
        # model = torch.jit.load('C:/Users/Sneha/Downloads/Cat_Dog-main/Detectron/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth').eval()
        # model.load_state_dict(torch.load('https: // download.pytorch.org / models / fasterrcnn_resnet50_fpn_coco - 258fb6c6.pth'))
        #   print(datasets)
          # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)



        # from detectron2.engine import DefaultTrainer

        # cfg = get_cfg()
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        # cfg.DATASETS.TRAIN = ("masktest2",)
        # cfg.DATASETS.TEST = ()
        # cfg.DATALOADER.NUM_WORKERS = 2
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        #     "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        # cfg.SOLVER.IMS_PER_BATCH = 2
        # cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
        # cfg.SOLVER.MAX_ITER = 1000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        # # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
        #
        # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        # trainer = DefaultTrainer(cfg)
        # trainer.resume_or_load(resume=True)
        # trainer.train()
        #
        # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/content/drive/MyDrive/detectron2/output/model_final.pth")
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
        # cfg.DATASETS.TEST = ("masktest2",)
        # predictor = DefaultPredictor(cfg)










        import pandas as pd
        df = pd.DataFrame(result.pandas().xyxy[0])
        bloodtype = df.groupby('name').count()
        pr = bloodtype.iloc[:, -1:]
        pri = pr.to_dict()
        # # return pri
        results=1
        if results == 1:
            prediction = pri
            return [{ "image" : prediction}]
        else:
            prediction = 'cat'
            return [{ "image" : prediction}]


