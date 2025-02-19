import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from  tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2


feature_List =np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filename .pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
#we are not training out model
model.trainable=False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img("Sample/1543.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)  ##for making batch of image
preprocessed_img = preprocess_input(expanded_img_array)
result= model.predict(preprocessed_img).flatten()
normalized_result = result/norm(result)

neighbors = NearestNeighbors(n_neighbors=6,algorithm="brute", metric='euclidean')
neighbors.fit(feature_List)

distances,indices = neighbors.kneighbors([normalized_result])
#print(indices)

for file in indices[0][1:6]:
    temp_img= cv2.imread(filenames[file])
    #cv2.imshow('output',cv2.resize(temp_img, (412,412)))
    #cv2.waitKey(0)
