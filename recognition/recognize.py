import numpy as np 
import pandas as pd
import cv2
import os
from keras.layers import Convolution2D, Activation, Flatten, Dense, BatchNormalization, Input, ZeroPadding2D, Rescaling
from keras.models import Model
from keras.utils import image_dataset_from_directory
import tensorflow as tf

# def FaceMaskModel(shape):
#     X_input = Input(shape)
#     padding_dimensions = (5, 5)
#     X = ZeroPadding2D(padding_dimensions)(X_input)
#     layers1 = 32
#     filter1 = (7, 7)
#     strides1 = (1, 1)
#     X = Convolution2D(layers1, filter1, strides=strides1, name="Convolution_1")(X)
#     X = BatchNormalization(axis=1, name="Batch_Normalization_1")(X)
#     X = Activation("relu")(X)
#     X = Flatten()(X)
#     X = Dense(1, activation="sigmoid", name="fully_connected_layer")(X)
#     face_mask_model = Model(inputs=X_input, outputs=X, name="Face_Mask_Classifier")
#     return face_mask_model
#
# def load_model(path_to_model):
#     from keras.models import load_model
#     model = load_model(path_to_model)
#     return model

# Используется для обучения и сохранения модели
if __name__ == "__main__":
    # Путь к данным
    train_dir = '/home/milchenko/programming/recognition-telegram-bot/data/Train/'
    test_dir = '/home/milchenko/programming/recognition-telegram-bot/data/Test/'
    validation_dir = '/home/milchenko/programming/recognition-telegram-bot/data/Validation/'

    # Define rescaling layer
    rescale = Rescaling(1./255)

    # Load train dataset with rescaling
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        batch_size=32,
        image_size=(224, 224),
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode='categorical',  # Assuming you have multiple classes
    )
    train_ds = train_ds.map(lambda x, y: (rescale(x), y))  # Apply rescaling

    # Load validation dataset with rescaling
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory=validation_dir,
        batch_size=32,
        image_size=(224, 224),
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode='categorical',  # Assuming you have multiple classes
    )
    validation_ds = validation_ds.map(lambda x, y: (rescale(x), y))  # Apply rescaling

    # Load test dataset with rescaling
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=test_dir,
        batch_size=32,
        image_size=(224, 224),
        label_mode='categorical',  # Assuming you have multiple classes
        shuffle=False,
    )
    test_ds = test_ds.map(lambda x, y: (rescale(x), y))  # Apply rescaling

# Define input layer
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))

# Define model architecture
    model = tf.keras.models.Sequential([
        input_layer,
        # Convolutional layer with 32 filters and 3x3 kernel size, using ReLU activation
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D(),
        # Convolutional layer with 32 filters and 3x3 kernel size, using ReLU activation
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D(),
        # Convolutional layer with 32 filters and 3x3 kernel size, using ReLU activation
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D(),
        # Flatten layer to convert 2D data to 1D
        tf.keras.layers.Flatten(),
        # Dropout Layer
        tf.keras.layers.Dropout(0.5),
        # Output layer with 1 units (for 10 classes) and softmax activation
        tf.keras.layers.Dense(2, activation='softmax'),
    ])


    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define early stopping callback
    from tensorflow.keras.callbacks import EarlyStopping
    # Define early stopping callback
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)



    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(test_ds) 

    # Print the test loss and accuracy
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

# Fit the model with callbacks
    history = model.fit(train_ds,
                        validation_data=validation_ds,
                        epochs=30,
                        callbacks=[early_stopping])

    model.save('mask_model.keras')
    # Код для загрузки и подготовки данных...
    # x_train = []
    # y_train = []
    # for folder in os.listdir(train_dir):
    #     if folder == "WithoutMask":
    #         val = 0
    #     else:
    #         val = 1
    #     for file_name in os.listdir(os.path.join(train_dir, folder)):
    #         folder_path = os.path.join(train_dir, folder)
    #         image_path = os.path.join(folder_path, file_name)
    #         image = cv2.imread(image_path)
    #         resized_image = cv2.resize(image, (64, 64))
    #         final_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    #         x_train.append(final_image)
    #         y_train.append(val)
    # x_train = np.array(x_train, dtype="float32")
    # y_train = np.array(y_train, dtype="int32")
    #
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
    #
    # face_mask_model = FaceMaskModel((64, 64, 3))
    # face_mask_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # face_mask_model.fit(x=x_train, y=y_train, epochs=30, batch_size=10, callbacks=[early_stopping])
    # 
    # Сохранение модели
