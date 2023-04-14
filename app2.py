from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load MNIST dataset
digits = load_digits()
X_digits = digits.images.reshape((len(digits.images), -1))
y_digits = digits.target

# Perform SVD on MNIST dataset
svd_digits = TruncatedSVD(n_components=30, random_state=42)
X_digits_svd = svd_digits.fit_transform(X_digits)

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Perform SVD on Iris dataset
svd_iris = TruncatedSVD(n_components=2, random_state=42)
X_iris_svd = svd_iris.fit_transform(X_iris)

# Train a Random Forest Classifier on MNIST dataset
rf_digits = RandomForestClassifier(n_estimators=100, random_state=42)
rf_digits.fit(X_digits_svd, y_digits)

# Train a Random Forest Classifier on Iris dataset
rf_iris = RandomForestClassifier(n_estimators=100, random_state=42)
rf_iris.fit(X_iris_svd, y_iris)


@app.route('/classify', methods=['POST'])
def classify():
    # Load input image from HTTP request
    img = request.files.get('image')
    img = Image.open(img).convert('L')
    img = img.resize((8, 8))
    img = np.array(img).reshape(1, -1)

    # Perform SVD on input image
    svd_img = TruncatedSVD(n_components=30, random_state=42)
    img_svd = svd_img.fit_transform(img)

    # Predict class of input image using Random Forest Classifier
    digits_pred = rf_digits.predict(img_svd)
    iris_pred = rf_iris.predict(img_svd)

    # Prepare response
    response = {'digits': int(digits_pred[0]), 'iris': int(iris_pred[0])}

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
