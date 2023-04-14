import matplotlib.pyplot as plt
import numpy as np
from functions.svd import SVD
import cv2

# Load the image
myFileGray='myPhotoGray.png'
img = cv2.imread(myFileGray)

# Apply SVD with increasing number of singular values
reconstructed_images=[]
for n in range(25, 256, 20):
    svd = SVD(n_components=n)
    img_transformed = svd.fit_transform(img)
    img_reconstructed = svd.transform(img_transformed)
    reconstructed_images.append(img_reconstructed)

  
def plot_images(images):
    fig, axs = plt.subplots(3, 4, figsize=(2, 2))
    axs = axs.ravel()
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap='gray',aspect='auto')
        axs[i].axis('off')
    plt.show()

plot_images(reconstructed_images)
