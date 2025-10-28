## digit_classifier

import numpy as np
import keras
from keras import layers
from keras.datasets import mnist
from keras.models import load_model
from PIL import Image
import pickle
import cv2  # opencv-python

# load model
if True:
    model_path = "digit_classifier4.keras"
    model = load_model(model_path)
else:
    # Define a simple neural network
    model = keras.Sequential([  # input_shape=(28, 28).
        keras.layers.Conv2D(12, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # new size: (26,26,12)
        keras.layers.Conv2D(12, (3, 3), activation='relu'),  # new size: (24,24,12)
        keras.layers.MaxPool2D(pool_size=(2, 2)),  # new size: (12,12,12)
        keras.layers.Conv2D(16, (3, 3), activation='relu'),  # new size: (10,10,16). #params=1744
        keras.layers.MaxPool2D(pool_size=(2, 2)),  #
        keras.layers.Flatten(),  # size = 5*5*16=400
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),  # params= 25,664
        keras.layers.Dense(10, activation='softmax')  # output layer # params= 650
    ])  # Total params: 29,486 (115.18 KB)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


# special font
for sFont in [1]:
    img_path= "C://Users//danie//Documents//Code//PythonScripts//sudoku//code//" + str(sFont) + ".png"
    img0 = Image.open(img_path)
    digit = np.array(img0)[:,:,0]
    digit = 255-digit
    digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
    img_array = digit_resized.astype("float32") / 255.0
    x_train = np.zeros((128,28,28))
    y_train = sFont*np.ones((128,))
    for k0 in range(128):
        img = img_array.copy()
        img = Image.fromarray(img)
        # random rotate
        angle = np.random.uniform(-5, 5)  # train_fonts_4
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        # random shift
        dx = np.random.randint(-5, 5)
        dy = np.random.randint(-1, 1)
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0)
        # Add noise
        if False:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 5, arr.shape)
            arr += noise
            # Clip to valid range
            img = np.clip(arr, 0, 255).astype(np.uint8)

        x_train[k0,:,:] = img
    tmp1 = x_train[0,:,:]
    model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)




train_dataset_name = r'C:\Users\danie\Documents\Code\PythonScripts\scratches\database\digits_generated_synthetic\train_fonts_7.pkl'
# load the train data
with open(train_dataset_name, "rb") as f:
    loaded_data = pickle.load(f)
x_train, y_train = loaded_data  # print(digits.data.shape)  # (1e4, 28, 28)
x_train = x_train.astype("float32") / 255.0  # Normalize to [0,1]
n_samples, n_features, n_features2 = x_train.shape


# === Step 1: Load MNIST dataset ===
if False:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    nTrain = len(x_train)
    nTest = len(x_test)
    # FILL SIZE
    for k0 in range(nTrain):
        img2 = x_train[k0]
        coords = cv2.findNonZero(img2)
        x, y, w, h = cv2.boundingRect(coords)
        digit = img2[y:y + h, x:x + w]
        digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        #img = Image.fromarray(digit_resized)
        x_train[k0] = digit_resized

    for k0 in range(nTest):
        img2 = x_test[k0]
        coords = cv2.findNonZero(img2)
        x, y, w, h = cv2.boundingRect(coords)
        digit = img2[y:y + h, x:x + w]
        digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        x_test[k0] = digit_resized

    # Normalize to [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0



# Train the model
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

# test:

# Save for reuse
model.save("digit_classifier4.keras")

print("Finished")

# === Step 4: Predict a single 28x28 image ===
def predict_image_digit(img_path, model_path="digit_classifier.keras"):
    model = load_model(model_path)

    # Load the image (grayscale, 28x28)
    img = Image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img_array = Image.img_to_array(img).astype("float32") / 255.0

    # Predict
    predictions = model.predict(img_array)
    digit = np.argmax(predictions)
    return digit

def predict_digit(digits, model_path="digit_classifier.keras"):
    model = load_model(model_path)

    # scale
    img_array = digits.astype("float32") / 255.0

    # Predict
    predictions = model.predict(img_array)
    digit = np.argmax(predictions)
    return digit
