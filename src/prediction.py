import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import LabelBinarizer


MODEL = 'trained_models/modelTest.h5'

# Load the data
test_set = pd.read_csv('data/sign_mnist_test.csv')
y = test_set['label']

y_test = test_set['label']
del test_set['label']

label_binarizer = LabelBinarizer()
y_test = label_binarizer.fit_transform(y_test)
x_test = test_set.values
#print(x_test.shape)

# Normalize the data
x_test = x_test / 255
# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_test = x_test.reshape(-1,28,28,1)

model = keras.models.load_model(MODEL)


def get_letter(pred):
    if pred < 9:
        return chr(65 + pred)
    else :
        return chr(66 + pred)

# test pour une image
ind = 807
image = x_test[ind]
image = image.reshape(1, 28, 28, 1)
#print(image.shape)


# classification
pred = model.predict(image)
pred = np.argmax(pred,axis = 1)
if pred >= 9:
    pred += 1
plt.imshow(image.reshape(28,28), cmap=plt.get_cmap('gray'))
plt.title("Prediction : " + get_letter(pred[0]) + " - Real : " + get_letter(y[ind]))
print("Classification: ", get_letter(pred[0]) + " - Real : " + get_letter(y[ind]))
plt.show()
""" 
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

 #Connaître  l'accuracy de notre modèle
countTrue =0
countFalse = 0

for i in range(4500) :
    print(i)
    ind = i
    image = x_test[ind]
    image = image.reshape(1, 28, 28, 1)

    # classification
    pred = model.predict(image)
    pred = np.argmax(pred,axis = 1)

    if(get_letter(pred[0]) != chr(65 + y[ind])) :
        countFalse+=1
    else :
        countTrue+=1

print("Count True : " + str(countTrue))
print("Count False : " + str(countFalse))
 """