import os
import sys
sys.path.insert(1, '../') 
from utils.loader import Loader
from utils.logger import Logger
import pandas as pd
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pickle

# Descargar la lista de stopwords si no está ya descargada
nltk.download('stopwords')

# Cargar las stopwords en español
spanish_stopwords = set(stopwords.words('spanish'))


class Embedder:
    
    verb_size = 200
    logger = Logger("Embedder", "NLP/log/embedder.log").get_logger()

    def custom_concat(self, row, cols):
        return ' '.join(
            f"no tiene {col_name}" if row.get(col_name) == "no tiene" or not isinstance(row.get(col_name), str) else str(row.get(col_name)) 
            for col_name in cols if col_name in row.index
        )

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s\d]', '', text) 
        tokens = text.split()
        # Aquí deberías definir `spanish_stopwords`, importándolos o definiéndolos si es necesario
        filtered_tokens = [token for token in tokens if token not in spanish_stopwords]
        return filtered_tokens

    def get_average_embedding(self, tokens, model):
        embeddings = [model[word] for word in tokens if word in model]
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

    def __embed(self, column):
        tokens = column.apply(self.preprocess_text)
        self.logger.info("Tokenization completed")
        model_w2v = Word2Vec(tokens, vector_size=self.verb_size, window=1, min_count=3, workers=8)
        return model_w2v.wv


    def __init__(self, verb_size, train = None):
        self.verb_size = verb_size
        #if train == None: return
        
        self.logger.info("Prepare Train Dataframe")
        
        descriptions = [col for col in train.columns if "description" in col]
        train['full_description'] = train.apply(self.custom_concat, axis=1, args=(descriptions,))
        
        self.logger.info("Description completed")

        filtered_columns = ["idx", "price", "km", "fuelType", "full_description"]
        train = train[filtered_columns]
        self.logger.info("Columns filtered")
        train.dropna(inplace=True)
        self.logger.info("NAs filtered")

        self.logger.info("Start trainset Embedding")
        self.word_vectors = self.__embed(train['full_description'])
        self.logger.info("Embedding finished")

    def embedding_process(self, column):
        tokens = column.apply(self.preprocess_text)
        return tokens.apply(lambda x: self.get_average_embedding(x, self.word_vectors))