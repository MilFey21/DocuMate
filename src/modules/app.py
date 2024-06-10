import os
import sys
import yaml
import time
import psutil
import shutil
import warnings
import argparse
import logging
from tqdm import tqdm
from typing import List, Any
import numpy as np
import pandas as pd

# import torch

from extract_embedding import FilePreprocessor
from ml_model import cos_sim_model, KNN
sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])  # Add parent folder to the Python path
from data.create_db import DB

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.backends else 'cpu')
supported_formats = ['.txt', '.docx', '.epub', '.pdf', '.csv', '.xls', '.xlsx', '.ppt', '.html']


def is_file_open(file_path: str) -> bool:
    """
    Function that checks if a file is open
    :param file_path: file path
    :return: True if file is open, else False
    """
    for proc in psutil.process_iter(['pid', 'open_files']):
        try:
            for item in proc.info['open_files']:
                if file_path == item.path:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def get_downloaded_files() -> List[str]:
    """
    Function that returns all files in Downloads folder
    """
    # Get the path to the Downloads folder for the current user
    downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    # List all files in the Downloads folder
    files = [downloads_path + '/' + f for f in os.listdir(downloads_path) if os.path.isfile(os.path.join(downloads_path, f))]
    return files


def run_app(config_path: str):
    config = yaml.safe_load(open(config_path))

    downloaded_files = get_downloaded_files()  # get files in the folder 'Downloads'

    # initialize database with existing files in target folders
    db = DB(config_path)
    files_db = db.get_data()

    # initialize and fit model
    model = KNN(
        config["knn"]["n_neighbors"],
        config["summarizer"]["model_name"],
        int(config["summarizer"]["max_length"]),
        config["vectorizer"]["sentence_transformer_name"],
        config["knn"]["use_nearest_centroids"]
    )
    model.fit(files_db)

    # initialize files preprocessor
    preprocessor = FilePreprocessor(
        summarizer_model_name=str(config['summarizer']['model_name']),
        summarizer_model_max_length=int(config['summarizer']['max_length']),
        st_model_name=str(config['vectorizer']['sentence_transformer_name'])
    )

    # iterate over each file from 'Downloads' folder 
    for downloaded_file_path in tqdm(downloaded_files):
        # if file is not open by user and its format is supported by Embedding Extractor
        # (is_file_open(downloaded_file_path) == False) and
        if os.path.splitext(downloaded_file_path)[1].lower() in supported_formats:
            logger.info(f"File {downloaded_file_path} was taken")
            file_embed = preprocessor(downloaded_file_path)  # get file embedding
            target_folder = model.predict(file_embed)
            # target_folder = cos_sim_model(target_folders, file_embed)  # get target folder prediction
            # print("pred_class_orig", target_folder)
            shutil.move(downloaded_file_path, target_folder)  # move file to defined folder
            logger.info(f"File {downloaded_file_path} was moved to to directory {target_folder}")
    # time.sleep(config['base']['sleep'])


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    run_app(config_path=args.config)
    # python src/modules/app.py --config=params.yaml
