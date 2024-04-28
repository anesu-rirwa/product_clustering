import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.utils import load_img
import pickle

# Function to extract features
def extract_features(file, model):
    try:
        img = load_img(file, target_size=(224,224))
        img = np.array(img) 
        reshaped_img = img.reshape(1,224,224,3) 
        imgx = preprocess_input(reshaped_img)
        features = model.predict(imgx)
        return features
    except Exception as e:
        st.write(f"Error processing {file}: {e}")
        return None

# Load VGG16 model
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Set the image directory path
IMAGE_DIR = "/home/anesu/Documents/assignment/AmazonProductImages"

# Main function
def main():
    st.title("Image Clustering App")

    # Extracting features
    products = []
    with os.scandir(IMAGE_DIR) as files:
        for file in files:
            if file.name.endswith('.jpg'):
                products.append(os.path.join(IMAGE_DIR, file.name))

    if products:
        data = {}
        for product in products:
            feat = extract_features(product, model)
            if feat is not None:
                data[product] = feat

        if data:
            # Save features
            pkl_path = os.path.join(IMAGE_DIR, 'features.pkl')
            with open(pkl_path, 'wb') as file:
                pickle.dump(data, file)
            st.write("Features extracted and saved.")

            # Reshape features
            filenames = np.array(list(data.keys()))
            feat = np.array(list(data.values()))
            feat = feat.reshape(-1,4096)

            # PCA and clustering
            pca = PCA(n_components=100, random_state=22)
            pca.fit(feat)
            x = pca.transform(feat)

            kmeans = KMeans(n_clusters=10, random_state=22)
            kmeans.fit(x)

            groups = {}
            for file, cluster in zip(filenames,kmeans.labels_):
                if cluster not in groups.keys():
                    groups[cluster] = []
                groups[cluster].append(file)

            st.write("Clustering complete.")

            # Display all clusters
            view_all_clusters(groups)

# Function to display all clusters
def view_all_clusters(groups):
    for cluster_id, files in groups.items():
        st.subheader(f"Cluster {cluster_id} Images")
        fig = plt.figure(figsize=(15, 15))
        for index, file in enumerate(files):
            plt.subplot(5, 6, index+1)
            img = load_img(file)
            img = np.array(img)
            plt.imshow(img)
            plt.axis('off')
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
