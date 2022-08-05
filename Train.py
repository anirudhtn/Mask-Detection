import os

DIRECTORY = os.getcwd() + "\\dataset"
CATEGORIES = ["with_mask", "without_mask"]

from tensorflow import keras

print("[PROGRESS] Loding images...")

data = []
labels = []

for category in CATEGORIES:
    category_path = os.path.join(DIRECTORY, category)
    for image in os.listdir(category_path):
        image_path = os.path.join(category_path, image)
        image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = keras.preprocessing.image.img_to_array(image)
        image = keras.applications.mobilenet_v2.preprocess_input(image)

        data.append(image)
        labels.append(category)

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = keras.utils.to_categorical(labels)

import numpy as np

data = np.array(data, dtype="float32")
labels = np.array(labels)

from sklearn.model_selection import train_test_split

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

from tensorflow import keras

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

aug = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

baseModel = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=keras.layers.Input(shape=(224, 224, 3)),
)

from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)

# Go for relu in nonlinear use cases, generally for images
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)

# Go for softmax or sigmoid as they're probability based 0 or 1
headModel = Dense(2, activation="softmax")(headModel)

model = keras.models.Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

from tensorflow.keras.optimizers import Adam

print("[PROGRESS] Compiling Model...")
# Go for adam optimizer as it is the goto in case of image use cases

opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("[PROGRESS] Training Head Model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
)

print("[PROGRESS] Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

from sklearn.metrics import classification_report

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

print("[PROGRESS] Saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# Visualize the model
import matplotlib.pyplot as plt

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
