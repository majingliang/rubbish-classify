from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
import matplotlib.image as mpimg
import cv2
im=cv2.imread('C:/Users/Administrator/Desktop/rubbissh/xxx.jpg')
plt.imshow(im)

im=cv2.resize(im, (150,150),interpolation = cv2.INTER_CUBIC)
im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im=np.reshape(im, [1, im.shape[0], im.shape[1], 3])
print (im.shape)
im = np.array(im)/255.0
model = load_model("C:/Users/Administrator/Desktop/rubbissh/cats_and_dogs_small_10.h5")
print(model.predict(im))
class_dict= {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
predicted_class = np.argmax(model.predict(im))
print(predicted_class)
label_map = dict((v,k) for k,v in class_dict.items())
predicted_label = label_map[predicted_class]
print(predicted_label)