import tensorflow as tf
from src.data_preprocessing import prepare_datasets

def main():
    data_path = 'NEU-DET'
    train_dataset, val_dataset, label_mapping = prepare_datasets(data_path)
    
    num_classes = len(label_mapping)
    
    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=10)
    
    # Save the model
    model.save('model\steel_defect_detection_model.h5')
    print("Model saved as 'steel_defect_detection_model.h5'")
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validation accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()