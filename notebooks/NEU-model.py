import tensorflow as tf
import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Paths
data_path = 'NEU-DET'
train_image_path = os.path.join(data_path, 'train', 'images')
train_annot_path = os.path.join(data_path, 'train', 'annotations')
validation_image_path = os.path.join(data_path, 'validation', 'images')

# Function to parse XML annotations
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append((class_name, xmin, ymin, xmax, ymax))
    return boxes

# Load image and annotation pairs
def load_data(image_dir, annot_dir):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = os.path.join(image_dir, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(class_dir, img_file)
                annot_path = os.path.join(annot_dir, img_file.replace('.jpg', '.xml'))
                if os.path.exists(annot_path):
                    boxes = parse_xml(annot_path)
                    if boxes:
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, (224, 224))  # Resize to match model input
                        images.append(img)
                        labels.append(class_indices[class_name])
    return np.array(images), np.array(labels), class_names

# Load and preprocess data
train_images, train_labels, class_names = load_data(train_image_path, train_annot_path)
val_images, val_labels, _ = load_data(validation_image_path, '')

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, stratify=train_labels)

# Normalize images
x_train = x_train / 255.0
x_val = x_val / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(x_val, y_val)
)

# Evaluate the model
loss, accuracy = model.evaluate(x_val, y_val)
print(f'Validation accuracy: {accuracy:.4f}')

# Save the model
model.save('neu_cls_model.h5')