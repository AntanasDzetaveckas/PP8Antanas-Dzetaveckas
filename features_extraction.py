from keras import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import time
import pickle
from numpy.linalg import norm
from model_finetuned import train_generator

# Load the custom model
model_finetuned = load_model('model-finetuned.keras')

# Print layer names
for layer in model_finetuned.layers:
    print(layer.name)

# Data generator
#datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#generator = datagen.flow_from_directory(root_dir, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode=None, shuffle=False)

# Feature extraction model
finetunned_cnn = Sequential()
finetunned_cnn.add(model_finetuned.get_layer('resnet50'))
finetunned_cnn.add(GlobalAveragePooling2D())

# Feature extraction
start_time = time.time()
feature_list_finetuned = finetunned_cnn.predict(train_generator)
end_time = time.time()

# Normalize features
for i, features_finetuned in enumerate(feature_list_finetuned):
    feature_list_finetuned[i] = features_finetuned / norm(features_finetuned)

# Reshape feature list
num_images = len(train_generator.filenames)
feature_list = feature_list_finetuned.reshape(num_images, -1)

# Save features
with open('features-caltech101-resnet-finetuned.pickle', 'wb') as f:
    pickle.dump(feature_list, f)

# Print stats
print(f"Num images = {len(train_generator.classes)}")
print(f"Shape of feature_list = {feature_list.shape}")
print(f"Time taken in sec = {end_time - start_time}")
print(f"Items per second = {len(train_generator.classes) / (end_time - start_time)}")
