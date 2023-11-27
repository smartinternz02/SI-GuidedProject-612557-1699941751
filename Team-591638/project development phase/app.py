# app.py

from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your pre-trained model
model = load_model('D:\\placements\\interships\\smartbridge data science\\project\\main\\UI\\sports_classification.h5')

# Function to preprocess the image before making predictions
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the pixel values to [0, 1]
    return img_array

# Function to make predictions
def predict_class(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']
        
        # Save the file to a temporary location
        file_path = 'C:\\Users\\heman\\OneDrive\\Desktop\\cricket1.jpeg'
        file.save(file_path)

        # Make a prediction on the uploaded image
        predicted_class = predict_class(file_path)

        # Redirect to the result page with the predicted class and image path
        return redirect(url_for('result', predicted_class=predicted_class, image_path=file_path))
    
    # If the request method is GET, render the main.html page
    return render_template('main.html')
# Define the list of class names
class_names = ['air hockey', 'amputee football', 'archery', 'arm wrestling', 'axe throwing', 'balance beam', 'barrel racing', 'baseball',
               'basketball', 'baton twirling', 'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding', 'bungee jumping',
               'canoe slalom', 'cheerleading', 'chuckwagon racing', 'cricket', 'croquet', 'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men',
               'figure skating pairs', 'figure skating women', 'fly fishing', 'football', 'formula 1 racing', 'frisbee', 'gaga', 'giant slalom',
               'golf', 'hammer throw', 'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping', 'horse racing', 'horseshoe pitching',
               'hurdles', 'hydroplane racing', 'ice climbing', 'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling',
               'luge', 'motorcycle racing', 'mushing', 'nascar racing', 'olympic wrestling', 'parallel bar', 'pole climbing', 'pole dancing', 'pole vault',
               'polo', 'pommel horse', 'rings', 'rock climbing', 'roller derby', 'rollerblade racing', 'rowing', 'rugby', 'sailboat racing', 'shot put',
               'shuffleboard', 'sidecar racing', 'ski jumping', 'sky surfing', 'skydiving', 'snowboarding', 'snowmobile racing', 'speed skating', 'steer wrestling',
               'sumo wrestling', 'surfing', 'swimming', 'table tennis', 'tennis', 'track bicycle', 'trapeze', 'tug of war', 'ultimate', 'uneven bars', 'volleyball',
               'water cycling', 'water polo', 'weightlifting', 'wheelchair basketball', 'wheelchair racing', 'wingsuit flying']

# Create a dictionary mapping class indices to class names
class_index_to_name = {i: name for i, name in enumerate(class_names)}

# ...

@app.route('/result/<int:predicted_class>/<path:image_path>')
def result(predicted_class, image_path):
    # Get the class name corresponding to the predicted class
    predicted_class_name = class_index_to_name.get(predicted_class, 'Unknown')

    # Render the result.html page with the predicted class and image path
    return render_template('result.html', predicted_class=predicted_class, predicted_class_name=predicted_class_name, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
