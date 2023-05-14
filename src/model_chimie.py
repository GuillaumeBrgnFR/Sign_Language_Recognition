import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer



# Load the data
train_set = pd.read_csv('data/sign_mnist_train.csv')

#Random state permet de reproduire les mêmes résultats
train_set, val_set = train_test_split(train_set, test_size = 0.2, random_state = 42) # split data



# donnees d'entree et de sortie
y_train = train_set['label']
y_val = val_set['label']
del train_set['label']
del val_set['label']

# preprocessing A VERIFIER
# Convertit les labels des classes en vecteur binaire (0...0,1,0...0) où 1 = classe de l'étiquette. 
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_val = label_binarizer.fit_transform(y_val)

x_train = train_set.values
x_val = val_set.values


# Normalize the data
x_train = x_train / 255
x_val = x_val / 255

# Reshaping the data from 1-D to 3-D as required through input by CNN's
# Réseau de neuronnes convolutions (CNN) requiert une forme appropriée
x_train = x_train.reshape(-1,28,28,1)
x_val = x_val.reshape(-1,28,28,1)

#Permet de générer de nouvelles images en appliquant des transformations aléatoires sur les données (images) - permet d'éviter le surapprentissage
dataGen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

dataGen.fit(x_train)

#Réduction du taux d'apprentissage lorsqu'une la performance du modèle ne s'améliore plus
lr_schedule = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=10**(-5))
# Paramètres : Si la précision de validation de s'améliore pas pendant 2 époques consécutives, le taux d'apprentissage sera réduit de 50% jusqu'à un minimum de 0.000001.
#early_stop = EarlyStopping(monitor='val_loss',patience=2)

tps1 = time.time()

# Model - CHIMIE
model = Sequential()

model.add(Conv2D(32 , kernel_size=3 , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) ))
model.add(Dropout(0.2))

model.add(Conv2D(64 , kernel_size=3 , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128 , kernel_size=3 , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))

#Transforme les données pour Dense
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dense(units = 24 , activation = 'softmax'))

# Compile the model
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# summarize layers
model.summary()

history = model.fit(dataGen.flow(x_train,y_train), epochs = 20 , validation_data = (x_val, y_val) , callbacks = [lr_schedule])

tps2 = time.time()

print("Temps d'execution : ", tps2 - tps1, " s")
# en minutes
print("Temps d'execution : ", (tps2 - tps1)/60, " min")

# save model and history
hist_df = pd.DataFrame(history.history) 
hist_json_file = 'trained_models/historyTest.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
model.save('trained_models/modelTest.h5')
