import os

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

#load the model
model = load_model("hotdogCNN2.h5")


#rescale pixel values from [0-255] to [0-1]
#ImageDataGenerator(rescale=1./255)


#classify
for file in os.listdir("data/validate/"):
    name = "data/validate/" + str(file)
    img = image.load_img(name, target_size= (300,300))


    pred_img = image.img_to_array(img)
    pred_img = np.expand_dims(pred_img, axis= 0)

    prediction = model.predict(pred_img)

    plt.imshow(img)
    plt.xlabel(prediction)

    plt.show()
