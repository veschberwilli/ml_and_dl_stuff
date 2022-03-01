"""
Flow:

1. load the model
2. get images
3. transform images
4. run inference
5. create exif tags

Another Tool: tag_finder
tool to query exif tags

Stretch Goal: train own face recognition model
"""
# from https://pytorch.org/hub/ultralytics_yolov5/

import torch
import glob
import os
import logging
import sqlalchemy as db
import pandas as pd
import argparse


# configuration for logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ])

class ImageTagger():
    def __init__(self) -> None:
    
        # params
        self.model_name = 'yolov5s'
        self.model_source = 'ultralytics/yolov5'
        self.path_to_imgs = os.path.join('pics')
        self.supported_img_formats = ['jpg']
        self.confidence_th = 0.5

        # start
        logging.info(f"Start Image Tagger...")

        # create db connection
        engine = db.create_engine('sqlite:///image_tags.sqlite')
        self.connection = engine.connect()

        # parse args
        self.run_parser()

        # ++++
        # RUN INFERENCE
        # ++++
        if self.config["run_inference"]:
            # load model
            self.load_model()

            # get images
            self.get_images()

            # run inference
            self.run_inference()

            # write tags to db
            self.write_tags_to_db()

        # ++++
        # QUERY BASED ON SEARCH TAG
        # ++++
        if self.config["query_tag"]:
            # search_tag=
            self.query_for_search_tag()

    def load_model(self) -> None:
        # load model
        logging.info(f"get the model {self.model_name}")
        self.model = torch.hub.load(self.model_source, self.model_name, pretrained=True)

    def get_images(self) -> None:
        # get images
        logging.info(f"Base Dir: {os.getcwd()}")
        logging.info(f"Locking for images in {self.path_to_imgs}")
        # TODO: supported_img_formats
        self.imgs = glob.glob(os.path.join(self.path_to_imgs, '*.jpg'))
        self.imgs_names = self.imgs.copy()

        # proceed if images found
        if len(self.imgs) == 0:
            raise Exception(f"No Images Found Under {self.path_to_imgs}.")

    def run_inference(self) -> None:
        # inference
        self.results = self.model(self.imgs)

        # Results
        #results.print() # prints an overview
        self.results.save()  # or .show() # saves images with detections

    def write_tags_to_db(self) -> None:
        # init Dataframe
        self.df = pd.DataFrame(columns=['path', 'img_name', 'detection'])

        # iterate over imgs
        for idx, img in enumerate(self.imgs_names):
            res = self.results.pandas().xyxy[idx][['confidence', 'name']]
            res_filtered = res[res['confidence'] >= self.confidence_th]['name'].to_list()
            logging.info(f"{img} --> {res_filtered}")
            
            basename = os.path.basename(img)
            full_path = os.path.realpath(img)
            # write to df
            for detection in res_filtered:
                data = {
                    'path': full_path,
                    'img_name': basename,
                    'detection': detection
                }
                self.df = self.df.append(data, ignore_index=True)

        # write df to db
        # TODO: not replace but complement
        self.df.to_sql('tags', self.connection, if_exists='replace')


    def query_for_search_tag(self, search_tag='horse') -> None:
        # look for specific tag
        
        # read db
        tags = pd.read_sql_table('tags', self.connection)

        # list query results
        logging.info(f"Searching for Tag: {search_tag}")

        results = tags[tags['detection'] == search_tag]['path']
        for res in results:
            logging.info(f"{res}")

    def run_parser(self):
        """
        definition of positional and optional arguments of this script
        """
        parser = argparse.ArgumentParser(
            description="""Script to Run Inference on Images and Query based on Tags. \n
            Example Call: python3 main.py -run_inference""",
            formatter_class=argparse.RawTextHelpFormatter,
            add_help=True,
        )

        control_setting = parser.add_argument_group("Parameters to Control the Steps")
        control_setting.add_argument("-run_inference", action="store_true", help="option to run inference.")
        control_setting.add_argument("-query_tag", action="store_true", help="option to query for one or more tags.")

        # read args
        args = parser.parse_args()
        self.config = {}
        self.config["run_inference"] = getattr(args, "run_inference")
        self.config["query_tag"] = getattr(args, "query_tag")


if __name__ == '__main__':
    ImageTagger()