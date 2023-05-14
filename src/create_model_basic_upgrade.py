import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load the data
train_set = pd.read_csv('data/sign_mnist_train.csv')
train_set, val_set = train_test_split(train_set, test_size = 0.2, random_state = 42) # split data


# donnees d'entree et de sortie
y_train = train_set['label']
y_val = val_set['label']
del train_set['label']
del val_set['label']

# preprocessing A VERIFIER
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_val = label_binarizer.fit_transform(y_val)

x_train = train_set.values
x_val = val_set.values


# Normalize the data
x_train = x_train / 255
x_val = x_val / 255

# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_train = x_train.reshape(-1,28,28,1)
x_val = x_val.reshape(-1,28,28,1)

tps1 = time.time()

# Model
#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(units = 24 , activation = 'softmax'))

# Compile the model
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# summarize layers
model.summary()

history = model.fit(x_train,y_train, epochs = 6 , validation_data = (x_val, y_val), batch_size = 128)

tps2 = time.time()

print("Temps d'execution : ", tps2 - tps1, " s")
# en minutes
print("Temps d'execution : ", (tps2 - tps1)/60, " min")

# save model and history
hist_df = pd.DataFrame(history.history) 
hist_json_file = 'trained_models/history_basic_upgrade.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
model.save('trained_models/model_basic_upgrade.h5')
