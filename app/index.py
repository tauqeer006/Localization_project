from flask import Flask, request, render_template, redirect, url_for
import os
import pickle
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

with open('models/trained_som.pkl', 'rb') as f:
    som = pickle.load(f)

original_array = np.load('models/thermal.npy', allow_pickle=True)
with open('models/encoded_features.pkl', 'rb') as f:
    encoded_array = pickle.load(f)
encoded_array = encoded_array[:9139]

latitude_min, latitude_max = -90, 90
longitude_min, longitude_max = -180, 180
som_width, som_height = som.get_weights().shape[:2]
latitudes = np.linspace(latitude_min, latitude_max, som_width)
longitudes = np.linspace(longitude_min, longitude_max, som_height)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def encode_image(image_path):
    img = imread(image_path)
    img_resized = resize(img, (224, 224), anti_aliasing=True)
    if img_resized.ndim == 2:  
        img_resized = np.stack([img_resized] * 3, axis=-1)
    img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))
    features = feature_extractor.predict(img_preprocessed)
    features_flattened = features.flatten()
    return features_flattened[:128]  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        uploaded_image_encoded = encode_image(filepath)
        winner_node = som.winner(uploaded_image_encoded)

        neighborhood_radius = 3
        neighboring_nodes = []
        for x in range(winner_node[0] - neighborhood_radius, winner_node[0] + neighborhood_radius + 1):
            for y in range(winner_node[1] - neighborhood_radius, winner_node[1] + neighborhood_radius + 1):
                if 0 <= x < som.get_weights().shape[0] and 0 <= y < som.get_weights().shape[1]:
                    neighboring_nodes.append((x, y))

        neighbor_data = []
        for i, (node) in enumerate(neighboring_nodes):
            x, y = node
            neighbor_image = som.get_weights()[x, y].flatten()
            dist = euclidean(uploaded_image_encoded, neighbor_image)

            closest_image_idx = np.argmin(np.linalg.norm(encoded_array - neighbor_image, axis=1))
            original_neighbor_image = original_array[closest_image_idx]

            neighbor_filename = f'neighbor_{i + 1}.png'
            neighbor_filepath = os.path.join(app.config['UPLOAD_FOLDER'], neighbor_filename)
            plt.imsave(neighbor_filepath, original_neighbor_image, cmap='gray')

            neighbor_data.append({
                'filepath': neighbor_filepath,
                'lat': latitudes[x],
                'lon': longitudes[y],
                'distance': dist
            })

        neighbor_data.sort(key=lambda x: x['distance'])
        closest_neighbors = neighbor_data[:10]

        return render_template('result.html', filepath=filepath, neighbors=closest_neighbors)


if __name__ == '__main__':
    app.run(debug=True)
