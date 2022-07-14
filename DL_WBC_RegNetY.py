import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from keras_cv_attention_models import resnet_family

batch_size = 30
size = (240, 240)

class_weight = {0: 0.485,1: 3.010,2: 1.743,3: 2.404,4: 0.877,5: 0.677}

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Train",
    labels="inferred",
    label_mode='categorical',
    class_names=["blast","promyelo","myelo","meta","band","seg"],
    image_size=size,
    batch_size=batch_size,
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Validation",
    labels="inferred",
    label_mode='categorical',
    class_names=["blast","promyelo","myelo","meta","band","seg"],
    image_size=size,
    batch_size=batch_size,
    shuffle=True
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

base_model = resnet_family.RegNetY040(input_shape=(240, 240, 3),num_classes=0)

base_model.trainable = True

scale_layer = keras.layers.Rescaling(scale=1 / 255)

data_augmentation = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation([-0.3,0.3]), layers.RandomZoom((0.3,-0.3),(0.3,-0.3)), layers.RandomContrast(0.3)])

inputs = keras.Input(shape=(240, 240, 3))
x = data_augmentation(inputs)
x = scale_layer(x)
x = base_model(x, training=True)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(544,activation = "relu")(x)
x = keras.layers.Dropout(0.3)(x)
#x = keras.layers.Dense(272,activation = "relu")(x)
#x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(6,activation = "softmax")(x)
model = keras.Model(inputs, outputs)

model.summary()

epochs = 100

callbacks = [keras.callbacks.ModelCheckpoint("RegNetY040_{epoch:04d}.h5",monitor='val_categorical_accuracy',verbose=1,save_best_only=True,mode='max'),]

model.compile(optimizer=keras.optimizers.Adam(5e-4),loss="categorical_crossentropy",metrics=["categorical_accuracy"])
history=model.fit(train_ds, epochs=epochs, batch_size=batch_size, class_weight=class_weight, callbacks=callbacks, validation_data=val_ds,shuffle=True)
np.save('RegNetY.npy',history.history)
