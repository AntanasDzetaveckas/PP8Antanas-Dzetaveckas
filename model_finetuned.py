from tqdm import tqdm, tqdm_notebook
import os
import random
import time
import math
import tensorflow
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from config import root_dir, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, NUM_CLASSES, TRAIN_SAMPLES

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2)

train_generator = train_datagen.flow_from_directory(root_dir,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=12345,
                                                    class_mode='categorical')


def model_maker():
    base_model = ResNet50(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    # base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max',
    # depth_multiplier=1, alpha=1)
    for layer in base_model.layers[:]:
        #layer.trainable = False
        layer.trainable = True
    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(
        custom_model)  # Resnet gives (1, 2048) if poolling applied and 7x7x2048 if not
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=predictions)


model_finetuned = model_maker()
model_finetuned.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(0.002),
                        metrics=['acc'])
#model_finetuned.fit(train_generator, epochs=5)

model_finetuned.save('./model-finetuned.keras')
