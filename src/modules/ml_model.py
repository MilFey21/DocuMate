import logging
import numpy as np
import pandas as pd
from typing import List

import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.backends else 'cpu')


def cos_sim_model(target_folders, file_embed):
    """
    Assign target folder based off of cosine similarity between file embedding and folder embedding
    :param target_folders:
    :param file_embed:
    :return:
    """
    logits = util.cos_sim(torch.Tensor(target_folders['content_emb']).to('cpu'), file_embed[np.newaxis, :]).ravel()
    probs = F.softmax(logits)
    pred_class_n = int(np.argmax(probs))
    pred_class_folder_path = target_folders.loc[pred_class_n, 'folder_path']
    return pred_class_folder_path


class KNN:
    def __init__(self, n_neighbors: int, summarizer_model_name: str,
                 summarizer_model_max_length: int, st_model_name: str, use_nc: bool = False):
        """
        Initialize KNN model
        :param n_neighbors: number of neighbors for KNN model
        :param summarizer_model_name: a name of model for text summarizer
        :param summarizer_model_max_length: maximum length of the summarized text for text summarizer
        :param st_model_name: Transformer model name fo text2embed model
        :param use_nc: indicator whether to use NearestCentroid model instead of KNN
        """
        self.n_neighbors = n_neighbors
        self.summarizer_model_name = summarizer_model_name
        self.summarizer_model_max_length = summarizer_model_max_length
        self.st_model_name = st_model_name
        self.label_encoder = LabelEncoder()
        if use_nc:
            self.neigh = NearestCentroid(metric='euclidean')
        else:
            self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Function fits classification model
        :param train_data: data to train model
        """
        X, y = pd.DataFrame(train_data['content_emb'].tolist()), train_data['folder_path']
        y = self.label_encoder.fit_transform(y)
        self.neigh.fit(X, y)

    def predict(self, X: np.array) -> str:
        """
        Function predicts class label, returning class label (string format)
        :param X: feature matrix
        :return: class label (string format)
        """
        pred_class = self.neigh.predict(X.reshape(1, -1))
        pred_class_orig = self.label_encoder.inverse_transform(pred_class)
        return pred_class_orig[0]
