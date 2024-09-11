import tensorflow as tf
import numpy as np
from src.data_preprocessing import prepare_datasets

def main():
    data_path = 'NEU-DET'
    _, val_dataset, label_mapping = prepare_datasets(data_path)
    
    num_classes = len(label_mapping)
    
    # Load the trained model
    model = tf.keras.models.load_model('model\steel_defect_detection_model.h5')
    
    # Make predictions on the validation dataset
    predictions = model.predict(val_dataset)
    
    # Convert predictions to label indices
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Get the true labels from the validation dataset
    true_labels = np.concatenate([label for _, label in val_dataset], axis=0)
    true_labels = np.argmax(true_labels, axis=1)  # Assuming one-hot encoded labels
    
    # Calculate accuracy manually
    accuracy = np.sum(predicted_labels == true_labels) / len(true_labels)
    print(f"Validation accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
