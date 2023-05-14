import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import random as rn

from sklearn.preprocessing import LabelBinarizer



def get_letter(pred):
    if pred < 9:
        return chr(65 + pred)
    else :
        return chr(66 + pred)


if __name__ == "__main__":

    # Load the model
    MODEL = 'trained_models/model.h5'
    model = keras.models.load_model(MODEL)

    # Load the data
    test_set_final = pd.read_csv('data/sign_mnist_test.csv')
    test_set = pd.read_csv('data/sign_mnist_test.csv')
    y = test_set_final['label']
    del test_set['label']
    x_test = test_set.values
    x_test = x_test / 255
    x_test = x_test.reshape(-1,28,28,1)
    print(x_test.shape)

    # enter a word
    word = input("Enter a word in capitalize: ")

    # split the word into a list
    list_word = list(word)

    # convert the list into a list of ascii code for test set
    list_ascii = [ord(i) for i in list_word]
    print(list_ascii)
    for i in range(len(list_ascii)):
        if list_ascii[i] > 73:
            list_ascii[i] -= 66
        else:
            list_ascii[i] -= 65 
    # select images
    index_df = []
    for i in list_ascii:
        print("i : ", i)
        # select images with the same label
        temp_df = test_set_final.loc[test_set_final['label'] == i]
        # select a random image
        try :
            temp_ind = rn.randint(0, len(temp_df)-1)
        except ValueError:
            temp_ind = 0
        i = temp_df.index[temp_ind]
        index_df.append(i)
    
    for i in index_df:
        print("index_df et lettre")
        print(i)
        print(get_letter(test_set_final.loc[i]['label']))

    
    fig, axes = plt.subplots(nrows=1, ncols=len(index_df), figsize=(10, 5))
    for i, ax in enumerate(axes):
        image = x_test[index_df[i]]
        image = image.reshape(1, 28, 28, 1)
        pred = model.predict(image)
        pred = np.argmax(pred,axis = 1)
        if pred >= 9: 
            pred += 1 
        ax.imshow(image.reshape(28,28), cmap=plt.get_cmap('gray'))
        ax.set_title(get_letter(pred[0]))
        ax.set_axis_off()
        print("HERE ",y[index_df[i]])
        print("Classification: ", get_letter(pred[0]), " - Real : ", get_letter(y[index_df[i]]))
    plt.show()
