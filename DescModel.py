import sys

sys.path.insert(1, '../') 
from utils.loader import Loader
from utils.logger import Logger
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization
import pickle

from gensim.models import Word2Vec, KeyedVectors

from .Embedder import Embedder

class DescModel:
    logger = Logger("Description_Model", "NLP/log/Description_Model.log").get_logger()
    log_dir = os.path.join("logs", "NLP")
    def __init__(self):
        # Intentar cargar el modelo o entrenar si no existe
        os.makedirs(self.log_dir, exist_ok=True)
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.load_model("NLP/models/model_2_text_km_to_price.keras")
    
    def load_model(self, path_modelo):
        """Carga el modelo desde un archivo si existe, de lo contrario retorna False para indicar el entrenamiento."""
        if os.path.isfile(path_modelo) and path_modelo.endswith('.keras'):
            try:
                self.model = tf.keras.models.load_model(path_modelo)
                self.logger.info("Modelo cargado exitosamente.")
                # Intenta cargar los escaladores también
                pkl_path = "NLP/models/desc_model.pkl"
                if os.path.exists(pkl_path):
                    self.embedder = self.load_embedder()
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                        self.scaler_embeddings = data['scaler_embeddings']
                        self.scaler_km = data['scaler_km']
                        self.scaler_y = data['scaler_y']
                return True
            except Exception as e:
                self.logger.error(f"Error al cargar el modelo: {e}")

                return False
        else:
            self.logger.error("El archivo no existe o no tiene la extensión '.keras'")

            self.logger.error(f"Entrenando el modelo")

            while (self.train_model() > 15): pass

            return False
    
    def save_model(self, base_path="NLP/models/"):
        """Guarda el modelo y los escaladores en el disco."""
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        # Guardar el modelo de TensorFlow
        model_path = os.path.join(base_path, "model_2_text_km_to_price.keras")
        self.model.save(model_path)
        self.save_embedder()
        # Guardar los escaladores y cualquier otro atributo necesario
        with open(os.path.join(base_path, "desc_model.pkl"), "wb") as f:
            pickle.dump({
                "scaler_embeddings": self.scaler_embeddings,
                "scaler_km": self.scaler_km,
                "scaler_y": self.scaler_y
            }, f)

        self.logger.info(f"Estado del modelo guardado en {base_path}")

    def preprocess_data(self, train):
        """Preprocesa los datos para obtener las descripciones embebidas y normaliza los valores."""
        descriptions = [col for col in train.columns if "description" in col]
        self.logger.info(f"Filtering text")
        train['full_description'] = train.apply(self.embedder.custom_concat, axis=1, args=(descriptions,))
        self.logger.info(f"Tokenizing text")
        train['embedding'] = self.embedder.embedding_process(train['full_description'])
        return train

    def split_and_scale_data(self):
        """Divide el dataset en entrenamiento y prueba, y escala las características."""
        self.logger.info("Loading train Data")
        train_df = Loader.load_train()
        self.logger.info("Loading test Data")
        test_df = Loader.load_test()
        
        self.embedder = Embedder(200, train=train_df)
        train_df = self.preprocess_data(train_df)
        test_df = self.preprocess_data(test_df)

        x_embeddings = np.stack(train_df["embedding"].values)
        x_km = train_df['km'].to_numpy().reshape(-1, 1)
        y = train_df['price'].to_numpy()

        self.scaler_embeddings = StandardScaler()
        x_embeddings_scaled = self.scaler_embeddings.fit_transform(x_embeddings)

        self.scaler_km = StandardScaler()
        x_km_scaled = self.scaler_km.fit_transform(x_km)

        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        return train_df, test_df, x_embeddings_scaled, x_km_scaled, y_scaled

    def build_model(self):
        """Construye el modelo de red neuronal."""
        input_embeddings = Input(shape=(200,), name='embeddings_input')
        input_km = Input(shape=(1,), name='km_input')

        # Procesamiento de embeddings
        x = Dense(200, activation='relu')(input_embeddings)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)

        # Procesamiento de 'km'
        km_processed = Dense(32, activation='relu')(input_km)

        # Combinar embeddings y 'km'
        combined = Concatenate()([x, km_processed])

        z = Dense(16, activation='relu'
                  )(combined)
        z = BatchNormalization()(z)
        z = Dense(1)(z)  # Capa de salida

        model = Model(inputs=[input_embeddings, input_km], outputs=z)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    def train_model(self):
        """Entrena el modelo con los datos dados."""
        train_df, test_df, x_embeddings_scaled, x_km_scaled, y_scaled = self.split_and_scale_data()

        self.model = self.build_model()

        self.logger.info("COMIENZA EL ENTRENAMIENTO...")
        self.model.fit(
            [x_embeddings_scaled, x_km_scaled],
            y_scaled,
            epochs=275,
            batch_size=32,
            verbose=True,
            callbacks=[self.tensorboard_callback]
        )
        self.logger.info("TERMINA EL ENTRENAMIENTO...")
        self.logger.info(self.model.summary())

        # Evaluación en datos de prueba
        new_embeddings_scaled = self.scaler_embeddings.transform(np.stack(test_df["embedding"].values))
        new_km_scaled = self.scaler_km.transform(test_df['km'].to_numpy().reshape(-1, 1))

        prediction = self.model.predict([new_embeddings_scaled, new_km_scaled]).flatten()
        prediction = self.scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()
        real_price = test_df['price'].to_numpy()

        result = pd.DataFrame({'Prediction': prediction, 'Real price': real_price, "km": test_df["km"]})

        diff = np.mean(abs((real_price - prediction) / real_price))
        self.logger.info(result)
        self.logger.info(f"Hay un MAPE de {diff * 100}%")
        self.save_model()
        return diff * 100

    def save_embedder(self, path="NLP/models/"):
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Guardar el modelo Word2Vec
        model_path = os.path.join(path, "word2vec.model")
        self.embedder.word_vectors.save(model_path)
        
        # Guardar los demás atributos que no son modelos o no son serializables directamente
        with open(os.path.join(path, "embedder.pkl"), "wb") as f:
            # Aquí puedes escoger qué otros atributos necesitas guardar, por ahora sólo `verb_size`
            pickle.dump({"verb_size": self.embedder.verb_size}, f)

        print(f"Modelo y configuración guardados en {path}")

    def load_embedder(self, path="NLP/models/"):
        # Cargar el modelo Word2Vec
        model_path = os.path.join(path, "word2vec.model")
        word_vectors = KeyedVectors.load(model_path)
        
        # Cargar los otros atributos guardados
        with open(os.path.join(path, "embedder.pkl"), "rb") as f:
            data = pickle.load(f)
        
        # Crear una instancia de `Embedder` (ajustar esto según cómo se debería manejar la reinicialización)
        embedder = Embedder(verb_size=data["verb_size"])  # Asumiendo que `train` puede ser un DataFrame vacío aquí
        embedder.word_vectors = word_vectors
        
        print(f"Modelo y configuración cargados desde {path}")
        return embedder
    
    def predict(self, x):
        x = self.preprocess_data(x)
        new_embeddings_scaled = self.scaler_embeddings.transform(np.stack(x["embedding"].values))
        new_km_scaled = self.scaler_km.transform(x['km'].to_numpy().reshape(-1, 1))

        prediction = self.model.predict([new_embeddings_scaled, new_km_scaled]).flatten()
        prediction = self.scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()

        return prediction
