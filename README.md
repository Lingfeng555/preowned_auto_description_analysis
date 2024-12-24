# 🚗 Car Price Prediction Models

This repository contains notebooks and scripts for predicting the price of second-hand cars using `km` and descriptive text fields. The workflow includes preprocessing text descriptions, generating embeddings, and building predictive models. To improve performance, **BETO**, a BERT-based model for Spanish, is utilized for contextual embeddings of text descriptions.

---

## 🗂️ Files and Techniques

### 1. **`Embedder.py`** 🛠️
A utility script for generating Word2Vec embeddings from car descriptions.
- **Techniques**:
  - **Text Preprocessing**: Tokenization, stopword removal (Spanish-specific), and normalization.
  - **Word2Vec**: Trains a Word2Vec model on car descriptions with a specified vector size (`verb_size`).
  - **Average Embeddings**: Generates a single embedding for each description by averaging word vectors.

**Usage**:
```python
from Embedder import Embedder

# Initialize the Embedder
embedder = Embedder(verb_size=200, train=train_df)

# Generate embeddings for a text column
embeddings = embedder.embedding_process(train_df['full_description'])
```

---

### 2. **`embedding.ipynb`** 📒
Demonstrates the initial process of creating and analyzing embeddings.
- **Techniques**:
  - Text cleaning and preprocessing.
  - Training a Word2Vec model and exploring the embeddings.
  - Basic visualization of the generated embeddings (e.g., using t-SNE or PCA).

---

### 3. **`embedding_v2.ipynb`** 📒
Refines the embedding process and integrates with a prediction model.
- **Techniques**:
  - Enhanced preprocessing of descriptive text fields.
  - Embedding integration with regression models (e.g., linear regression or decision trees).
  - Performance evaluation using metrics like Mean Squared Error (MSE).

---

### 4. **`embedding_v3.ipynb`** 📒
Explores hyperparameter tuning and advanced models for price prediction.
- **Techniques**:
  - Fine-tuning Word2Vec parameters (e.g., vector size, window size).
  - Leveraging BETO embeddings for contextual understanding of text descriptions.
  - Experimenting with various machine learning models (e.g., random forests, gradient boosting).
  - Evaluating the impact of embeddings on model performance.

---

### 5. **`embedding_v4.ipynb`** 📒
Finalizes the workflow with optimizations and expanded analysis.
- **Techniques**:
  - Combining BETO embeddings with numerical features (`km`).
  - Optimizing regression models for better predictions.
  - Detailed error analysis and visualization.

---

## 💡 Applied NLP Techniques

1. **Text Preprocessing**:
   - Lowercasing and removing special characters.
   - Tokenizing text into words.
   - Filtering out Spanish stopwords.

2. **Embedding Creation**:
   - **Word2Vec** embeddings trained specifically on descriptive text fields.
   - **BETO Contextual Embeddings**: BETO is used to extract richer, contextual word embeddings that improve the performance of downstream tasks.
   - Averaging embeddings to represent entire descriptions.

3. **Model Integration**:
   - Combining embeddings (Word2Vec + BETO) with numerical features for predictive modeling.
   - Exploring various regression techniques to predict car prices.

---

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
- Use `Embedder.py` to preprocess and generate embeddings.
- Experiment with the notebooks to refine and evaluate your models.