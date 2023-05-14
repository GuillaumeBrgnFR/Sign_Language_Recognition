import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import keras

from sklearn.metrics import confusion_matrix, classification_report


MODEL = 'trained_models/model_basic_upgrade.h5'
HISTORY = 'trained_models/history_basic_upgrade.json'


test_set = pd.read_csv('data/sign_mnist_test.csv')
y = test_set['label']

y_test = test_set['label']
del test_set['label']

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_test = label_binarizer.fit_transform(y_test)
x_test = test_set.values

# Normalize the data
x_test = x_test / 255

# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_test = x_test.reshape(-1,28,28,1)

model = keras.models.load_model(MODEL)


print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

history = pd.read_json(HISTORY)
epochs = [i for i in range(6)]  # 20 epochs de base mais 6 pour le basic suffit
fig , ax = plt.subplots(1,2)
train_acc = history['accuracy']
train_loss = history['loss']
val_acc = history['val_accuracy']
val_loss = history['val_loss']
fig.set_size_inches(16,9)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Validation Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

predictions = model.predict(x_test)
predictions = np.argmax(predictions,axis = 1)
for i in range(len(predictions)):
    if(predictions[i] >= 9):
        predictions[i] += 1
predictions[:5]  

classes = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(y, predictions, target_names = classes))

 #Affichage de la matrice de confusion (pour comparaison avec résultat de Thibault)
cm = confusion_matrix(y,predictions)
cm = pd.DataFrame(cm , index = [i for i in range(25) if i != 9] , columns = [i for i in range(25) if i != 9])
plt.figure(figsize = (15,15))
sns.heatmap(cm, linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
plt.show() 