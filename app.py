
import pickle
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.neighbors import NearestNeighbors

st.title("Fashion Recommendation System")

features = np.array(pickle.load(open('Images_features.pkl', 'rb')))  # Özellikleri yükler
filenames = pickle.load(open('filenames.pkl', 'rb'))  # Görüntü yollarını yükler

@st.cache_resource
def load_model():
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    resnet.trainable = False
    return Sequential([resnet, GlobalMaxPooling2D()])

model = load_model()  # ResNet50 modelini hazırlar

uploaded = st.file_uploader("Fotoğraf yükleyin", type=['jpg', 'png', 'jpeg'])

if uploaded:
    st.image(uploaded, width=250)  # Yüklenen fotoğrafı gösterir
    img = Image.open(uploaded).convert('RGB').resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    vector = model.predict(preprocess_input(arr), verbose=0).flatten()
    vector = vector / np.linalg.norm(vector)  # Normalizasyon

    knn = NearestNeighbors(n_neighbors=min(5, len(features)))
    knn.fit(features)
    indices = knn.kneighbors([vector], return_distance=False)[0]  # Benzerleri bulur

    st.subheader("Benzer Ürünler")
    cols = st.columns(len(indices))
    for i, idx in enumerate(indices):
        cols[i].image(filenames[idx], use_container_width=True)  # Önerileri gösterir
