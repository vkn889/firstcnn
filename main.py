# Simple CNN Tutorial - Complete Implementation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# Step 1: Import Required Libraries
print("Step 1: Libraries imported successfully!")

# Step 2: Prepare Your Data
print("\nStep 2: Loading and preparing data...")

# Load CIFAR-10 dataset (built-in dataset)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to 0-1 range
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize some sample images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(f'Class: {class_names[np.argmax(y_train[i])]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Step 3: Design the CNN Architecture
print("\nStep 3: Building CNN model...")

model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(num_classes, activation='softmax')
])

# Step 4: Compile the Model
print("\nStep 4: Compiling the model...")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# View model architecture
model.summary()

# Step 5: Train the Model
print("\nStep 5: Training the model...")

# Set training parameters
batch_size = 32
epochs = 10

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=1
)

# Step 6: Evaluate and Test
print("\nStep 6: Evaluating the model...")

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 7: Make Predictions
print("\nStep 7: Making predictions...")

# Make predictions on test data
predictions = model.predict(x_test[:9])
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test[:9], axis=1)

# Visualize predictions
plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i])

    predicted_class = class_names[predicted_classes[i]]
    actual_class = class_names[actual_classes[i]]
    confidence = np.max(predictions[i])

    # Color the title green if correct, red if wrong
    color = 'green' if predicted_classes[i] == actual_classes[i] else 'red'
    plt.title(f'Predicted: {predicted_class}\nActual: {actual_class}\nConfidence: {confidence:.2f}',
              color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()

print("Predicted classes:", [class_names[i] for i in predicted_classes])
print("Actual classes:", [class_names[i] for i in actual_classes])

# Save the model
print("\nSaving the model...")
model.save('simple_cnn_model.h5')
print("Model saved as 'simple_cnn_model.h5'")

print("\n🎉 CNN Tutorial Complete!")
print(f"Final Test Accuracy: {test_accuracy:.2%}")

# Bonus: Function to predict on a single image
def predict_single_image(model, image, class_names):
    """
    Predict class for a single image
    """
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_names[predicted_class], confidence

# Example of using the prediction function
print("\nExample single prediction:")
test_image = x_test[0]
predicted_class, confidence = predict_single_image(model, test_image, class_names)
actual_class = class_names[np.argmax(y_test[0])]

print(f"Predicted: {predicted_class} (confidence: {confidence:.2f})")
print(f"Actual: {actual_class}")

# Additional model information
print(f"\nModel Summary:")
print(f"Total parameters: {model.count_params():,}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"Number of layers: {len(model.layers)}")

print("\n✅ All steps completed successfully!")
