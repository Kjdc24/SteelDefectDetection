import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(data_path):
    image_path = os.path.join(data_path, 'train', 'images')
    annot_path = os.path.join(data_path, 'train', 'annotations')
    
    images = []
    labels = []
    
    for label_folder in os.listdir(image_path):
        folder_path = os.path.join(image_path, label_folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img_path = os.path.join(folder_path, img_file)
                    img = load_img(img_path, target_size=(224, 224))
                    img = img_to_array(img)
                    
                    annot_file = img_file.replace('.jpg', '.xml').replace('.png', '.xml')
                    annot_file_path = os.path.join(annot_path, annot_file)
                    if os.path.exists(annot_file_path):
                        tree = ET.parse(annot_file_path)
                        root = tree.getroot()
                        label = root.find('object').find('name').text
                        images.append(img)
                        labels.append(label)
    
    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    
    return images, labels

def preprocess_data(images, labels):
    label_mapping = {label: idx for idx, label in enumerate(set(labels))}
    labels = [label_mapping[label] for label in labels]
    images = np.array(images)  # Convert to numpy array for preprocessing
    images = tf.keras.applications.resnet50.preprocess_input(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_mapping))
    
    return images, labels, label_mapping

def prepare_datasets(data_path):
    images, labels = load_data(data_path)
    images, labels, label_mapping = preprocess_data(images, labels)
    
    # Check if lengths are consistent
    if len(images) != len(labels):
        raise ValueError(f"Inconsistent data lengths: {len(images)} images, {len(labels)} labels.")
    
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, label_mapping