from flask import Flask, request
from sklearn.cluster import KMeans
from flask_restful import Resource, Api
import cv2
import numpy as np
import requests
from flask_cors import CORS


app = Flask(__name__)
api = Api(app)

# Enable CORS for all routes
CORS(app)

# Calculate and return the histogram of the image
def calculate_histogram(image):
    # RGB -> HSV.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histograms = []
    for i in range(3):
        # Hist prend la valeur de l'histogramme de hsv sur la canal i.
        hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
        histograms.append(hist.flatten().tolist())
    return histograms

# Calculate and return moments of the image
def calculate_moments(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the moments of the grayscale image
    moments = cv2.moments(gray_image)

    # Calculate the centroid (X, Y) of the object
    X = int(moments["m10"] / moments["m00"])
    Y = int(moments["m01"] / moments["m00"])

    return (X, Y)

def find_dominant_colors(image, num_colors=3):
    # convert the image to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply K-Means clustering
    clt = KMeans(n_clusters=num_colors, n_init=10)
    clt.fit(image.reshape(-1, 3))
    dominant_colors = np.uint8(clt.cluster_centers_)
    return dominant_colors.tolist()

class HistogramDescriptor(Resource):
    def get(self):
        image_url = request.args.get('image_url')

        if image_url:
            image = fetch_image(image_url)
            if image is not None:
                histogram = calculate_histogram(image)
                return {"histogram": histogram}
            else:
                return {"error": "Failed to decode the image"}, 500
        else:
            return {"error": "Image URL not provided"}, 400

class MomentsDescriptor(Resource):
    def get(self):
        image_url = request.args.get('image_url')

        if image_url:
            image = fetch_image(image_url)
            if image is not None:
                moments = calculate_moments(image)
                return {"moments": moments}
            else:
                return {"error": "Failed to decode the image"}, 500
        else:
            return {"error": "Image URL not provided"}, 400

class DominantColorsDescriptor(Resource):
    def get(self):
        image_url = request.args.get('image_url')
        num_colors = request.args.get('num_colors')

        if image_url :
            image = fetch_image(image_url)
            if image is not None:
                if num_colors:
                    dominant_colors = find_dominant_colors(image, int(num_colors))
                else:
                    dominant_colors = find_dominant_colors(image)
                return {"dominant_colors": dominant_colors}
            else:
                return {"error": "Failed to decode the image"}, 500
        else:
            return {"error": "Image URL or numbre of colors not provided"}, 400

# Helper function to fetch and decode the image from the URL
def fetch_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_bytes = response.content
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    return None

# Add the resources to Flask application using api.add_resource
api.add_resource(HistogramDescriptor, '/histogram')
api.add_resource(MomentsDescriptor, '/moments')
api.add_resource(DominantColorsDescriptor, '/dominant_colors')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=true)
