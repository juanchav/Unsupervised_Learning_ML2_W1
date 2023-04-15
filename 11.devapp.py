import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from functions.pca import PCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import random

# Load MNIST dataset from local file with allow_pickle=True
mnist_data = np.load('mnist_dataset.npz', allow_pickle=True)
X = mnist_data['X']
y = mnist_data['y']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA(n_components=2)
# fit the data
pca.fit(X_train)
# transform the data using the PCA object
X_train_pca = pca.transform(X_train)

classifier = LogisticRegression(max_iter=10000)

classifier.fit(X_train_pca, y_train)

# Perform PCA on testing data
X_test_pca = pca.transform(X_test)

number_to_predict=random.randrange(1,len(X_test_pca))
print(number_to_predict)


#img_tra=pca.inverse_transform(X_test_pca[number_to_predict-1:number_to_predict])
img_tra=pca.inverse_transform(X_test_pca[number_to_predict:number_to_predict,])
print('Este rango que es',X_test_pca[number_to_predict-1:number_to_predict])

# Make predictions on testing data
#y_pred = classifier.predict(X_test_pca[:number_to_predict])
y_pred = classifier.predict(X_test_pca[number_to_predict-1:number_to_predict])

# Calculate accuracy of the classifierq
print(y_pred)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
