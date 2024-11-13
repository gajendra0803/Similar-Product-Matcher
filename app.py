import tensorflow
import h5py
import numpy as np
from pyexpat import features
from  tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy .linalg import  norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
#we are not training out model
model.trainable=False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)  ##for making batch of image
    preprocessed_img = preprocess_input(expanded_img_array)
    result= model.predict(preprocessed_img).flatten()
    normalized_result = result/norm(result)

    return normalized_result

# Using a generator to avoid memory overflow
def image_generator(filenames, model):
    for file in filenames:
        yield extract_features(file, model)


filenames = [os.path.join('images', file) for file in os.listdir('images')]

feature_List = []
for feature in tqdm(image_generator(filenames, model), total=len(filenames)):
    feature_List.append(feature)

pickle.dump(feature_List, open('embeddings.pkl','wb' ))
pickle.dump(filenames, open('filename .pkl', 'wb'))

##print(np.array(feature_List).shape)
##print(os.listdir('images')) ## for listing all images in images directory