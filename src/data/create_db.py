# -*- coding: utf-8 -*-
# import click
import os
import sys
import yaml
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Union

# sys.path.append('..')  # Add parent folder to the Python path
sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])  # Add parent folder to the Python path
from modules.extract_embedding import FilePreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DB:
    def __init__(
            self,
            config_path: str,
            db_path: str = 'db.pkl',
    ):
        self.db_path = db_path
        self.files = self._initialize(config_path)  # call db initialization/actualization

    @staticmethod
    def get_files_in_folder(folder_path) -> List[str]:
        """
        Get paths to the files in the given folder of one of the following formats: '.txt', '.docx', '.epub', '.pdf',
        '.csv', '.xls', '.xlsx', '.ppt', '.html'
        :param folder_path: folder path, files of which we would add
        :return: list of file paths
        """
        allowed_formats = ['.txt', '.docx', '.epub', '.pdf', '.csv', '.xls', '.xlsx', '.ppt', '.html', '.dmate']
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in allowed_formats]
        return files

    @staticmethod
    def get_target_folders(root_folder: str) -> list[str | None]:
        """
        Get all the folders from the passed root_folder that have .dmate files in them and thus should be indexed
        :param root_folder: root folder that defines the algorithm's reach
        :param st_model_name: SentenceTransformer model name for "Text 2 Embedding" task
        :return: pd.DataFrame with 3 columns: folder_path (path to the folder where .dmate lies), content (the content of
        .dmate file), content_emb (embedding representation of the content of the .dmate file)
        """
        folder_paths = []
        for folder_path, _, files in os.walk(root_folder):
            dmate_file = [f for f in files if f.endswith('.dmate')]
            if dmate_file:
                folder_paths.append(folder_path)
        return folder_paths

    def _initialize(self, config_path: str) -> Union[pd.DataFrame | None]:
        """
        Initialize or actualize database
        :param config_path: path to config file
        :return: pd.DataFrame with info on files in target folders or None
        """
        config = yaml.safe_load(open(config_path))

        try:
            files_db = pd.read_pickle(self.db_path)
            files_db_path = files_db["file_path"].to_list()
            logger.info(f"Loaded {len(files_db)} files from {self.db_path}")
        except FileNotFoundError:
            files_db = pd.DataFrame()
            files_db_path = []
            logger.info("No prior DB was located")

        files_new = []
        target_folders = self.get_target_folders(config["base"]["root_folder"])
        preprocessor = FilePreprocessor(
            summarizer_model_name=config["summarizer"]["model_name"],
            summarizer_model_max_length=int(config["summarizer"]["max_length"]),
            st_model_name=config["vectorizer"]["sentence_transformer_name"]
        )
        # iterate over folders and files in the target folders
        for folder_path in tqdm(target_folders):
            folder_files = self.get_files_in_folder(folder_path)
            for file_path in folder_files:
                if file_path not in files_db_path:
                    full_file_path = folder_path + '/' + file_path
                    # print(full_file_path)
                    file_embed = preprocessor(file_path=full_file_path)
                    files_new.append(
                        {
                            'folder_path': folder_path,
                            'file_path': file_path,
                            'content_emb': file_embed
                        }
                    )
        files = pd.concat([files_db, pd.DataFrame(files_new)], ignore_index=True)

        # serialize files
        with open(self.db_path, "wb") as f:
            files.to_pickle(f)
        return files

    def get_data(self) -> pd.DataFrame:
        """
        Read and return Data with files in target folders
        :return: pd.DataFrame
        """
        return self.files


if __name__ == '__main__':
    # # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]
    #
    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    db = DB(config_path=args.config)
    files_db = db.get_data()
