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
    images = np.array(images)
    images = tf.keras.applications.resnet50.preprocess_input(images)  # Normalize input for ResNet50
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_mapping))
    
    return images, labels, label_mapping

def prepare_datasets(data_path):
    images, labels = load_data(data_path)
    images, labels, label_mapping = preprocess_data(images, labels)
    
    if len(images) != len(labels):
        raise ValueError(f"Inconsistent data lengths: {len(images)} images, {len(labels)} labels.")
    
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, label_mapping

def main():
    data_path = 'NEU-DET'
    train_dataset, val_dataset, label_mapping = prepare_datasets(data_path)
    
    num_classes = len(label_mapping)
    
    # Load ResNet50 and freeze layers
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze ResNet50 layers
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    
    # Save the model (fixing Windows path issue)
    model_save_path = os.path.join("model", "steel_defect_detection_model.h5")
    model.save(model_save_path)
    print(f"Model saved as '{model_save_path}'")
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validation accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()
