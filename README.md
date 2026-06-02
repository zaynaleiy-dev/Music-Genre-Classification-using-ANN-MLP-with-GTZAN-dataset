#  Music Genre Classification using ANN (MLP)

##  Project Overview
This project implements a **Multi-Layer Perceptron (MLP)** neural network to classify music tracks into 10 genres using the **GTZAN dataset**. It applies supervised learning, multiclass classification, activation functions, backpropagation, and overfitting control techniques.

##  Dataset
- Source: GTZAN Genre Collection
- Features: 57 pre-extracted audio features (MFCCs, chroma, spectral centroid, tempo, etc.)
- Labels: 10 genres (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock)

##  Methodology
1. Data preprocessing (LabelEncoder, StandardScaler, train-test split)
2. ANN training with MLPClassifier (ReLU, Adam optimizer, early stopping, L2 regularization)
3. Evaluation (accuracy, precision, recall, F1-score, confusion matrix)
4. Comparison with Logistic Regression baseline

##  Deployment
- **Streamlit App**: Interactive web app for genre prediction
- **Saved Models**: `.pkl` files for ANN, encoder, and scaler

##  Results
- Target accuracy: 80%+
- Confusion matrix and classification report included
- ANN outperforms Logistic Regression baseline

##  Deliverables
- LinkedIn Post: [add your LinkedIn post link here]
- GitHub Repo: https://github.com/zaynaleiy-dev/Music-Genre-Classification-using-ANN-MLP-with-GTZAN-dataset
- Registration No: 23108374
- Name: Zain Ali
- Deployed App: [https://music-genre-classification-using-ann-mlp-with-gtzan-dataset-2g.streamlit.app/]



