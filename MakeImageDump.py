# Import necessary library
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Set image category and paths and other variables
DATADIR = "images"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50


# For showing the first image make the value 1 or 0 for not showing
if(0):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # paths to drives from different category
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            plt.imshow(img_array, cmap="gray")
            plt.show()
            break
        break

    print(img_array)

# For resizing the image              hight     width
#new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#plt.imshow(new_array, cmap="gray")
#plt.show()

# Training Data
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))

import random
random.shuffle(training_data)


X = []  # Feature vector
y = []  # Lables

for feature, label in training_data:
    X.append(feature)
    y.append(label)

# Convert to numpy array
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Make pickle dump
import pickle
# Features are going to be dump called 'X.pickle'
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

# Labels are going to be dump called 'y.pickle'
pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

# For loading dump and printing the values for first 10 columns
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
#print(X[:10])

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
#print(y[:10])
