# -*- coding: utf-8 -*-
""" Implementation of SentenceBERT encoder.

@Author: Evan Dufraisse
@Date: Sun Oct 13 2024
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2024 CEA - LASTI
"""

import numpy as np
from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F



class SentenceBERTEncoder:

    def __init__(self, path_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Constructor of SentenceBERTEncoder.

        Args:
        path_model (str): Path to the SentenceBERT model.
        """

        self.model = AutoModel.from_pretrained(path_model)
        self.tokenizer = AutoTokenizer.from_pretrained(path_model)

        self.model.eval()
        

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences: List[str])-> np.ndarray:
        """
        Encode sentences into embeddings using SentenceTransformer model.

        Args:
        sentences (List[str]): List of sentences to encode.
        model (PreTrainedModel): Transformer model to use for encoding.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for encoding.

        Returns:
        np.ndarray: Array of embeddings for the input sentences.
        """

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.numpy()