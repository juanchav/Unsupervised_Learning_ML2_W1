from functions.svd import SVD
import cv2
import numpy as np

myFileGray='../pictureFace/myPhotoGray.png'
myPhotoGray=cv2.imread(myFileGray)

#cv2.imread('../pictureFace')

svd=SVD(n_components=10)
from PIL import Image
#U, s, Vt =svd.fit(myPhotoGray)
U, s, Vt = np.linalg.svd(myPhotoGray)
#U, s, Vt =svd.fit_transform(myPhoto)

#reconstructed_img = Image.fromarray(myPhoto_transformed.astype(np.uint8))
n_components = 3
reconstructed_gray = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
reconstructed_img = cv2.cvtColor(reconstructed_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

