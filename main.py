import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from functions.pca import PCA
from functions_rushter.pca_rushter import PCA as PCA_RUSHTER
from functions.svd import SVD
from functions.tsne import TSNE

csv=r'C:\Code\Spaces\Python\Proyectos\UdeA\Machine learning 2\datasets\01_AutoPrices.csv'
df=pd.read_csv(csv,encoding='utf-8')
df.fillna(0,inplace=True)
X=df.select_dtypes(['float64','int64'])
X=X.drop(['price'],axis=1)
y=df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1111)


pca=PCA(n_components=6)
pca_rushter=PCA_RUSHTER(n_components=6)
svd=SVD(n_components=6)
tsne=TSNE(n_components=6)

#pca.fit(X)
#X_transformed = pca.transform(X)

pca_rushter.fit(X)
X_transformed = pca_rushter.transform(X)

#svd.fit(X)
#X_transformed = svd.transform(X)

#tsne.fit(X)
#X_transformed = tsne.transform(X)


# plot the transformed data


plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
plt.colorbar()
plt.show()




		




