from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .models import UploadedImage
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define paths for training data
train_folder_yes = 'F:/WEB ENGINEERING/django/tumor_detection/detection/model/brain/yes'
train_folder_no = 'F:/WEB ENGINEERING/django/tumor_detection/detection/model/brain/no'
img_size = 128
conversion_factor = 0.264  # Conversion factor for mm²

# Image processing function
def process_image(image_path, is_tumor):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency
    image_resized = cv2.resize(image, (img_size, img_size))  # Resize for model input
    image_array = img_to_array(image_resized) / 255.0  # Normalize the image to [0, 1]

    area_pixels = 0
    area_mm2 = 0

    if is_tumor:
        # Convert to grayscale for tumor detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply thresholding (simple binary thresholding)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the tumor (largest contour assumed to be the tumor)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)  # Get the largest contour
            area_pixels = cv2.contourArea(contour)  # Area in pixels
            area_mm2 = area_pixels * (conversion_factor ** 2)  # Convert to mm²

    return image_array, area_pixels, area_mm2

# Prepare training data (images and labels)
images = []
labels = []
tumor_areas_pixels = []
tumor_areas_mm2 = []

# Process "yes" folder (tumor present)
for filename in os.listdir(train_folder_yes):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(train_folder_yes, filename)
        image_array, area_pixels, area_mm2 = process_image(image_path, is_tumor=True)
        images.append(image_array)
        labels.append('yes')
        tumor_areas_pixels.append(area_pixels)
        tumor_areas_mm2.append(area_mm2)

# Process "no" folder (tumor absent)
for filename in os.listdir(train_folder_no):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(train_folder_no, filename)
        image_array, _, _ = process_image(image_path, is_tumor=False)
        images.append(image_array)
        labels.append('no')
        tumor_areas_pixels.append(0)  # No tumor area
        tumor_areas_mm2.append(0)

# Convert images to numpy array
X = np.array(images)

# Encode labels (yes/no)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = to_categorical(y)  # One-hot encode labels

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: 'yes' and 'no'
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {accuracy:.2f}")

# Upload and process image
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_image_url = fs.url(filename)

        # Get the path to the uploaded image
        image_path = os.path.join(fs.location, filename)

        # Process the image for prediction
        image_array, area_pixels, area_mm2 = process_image(image_path, is_tumor=True)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make prediction using the trained model
        prediction = model.predict(image_array)
        class_idx = np.argmax(prediction)  # Get the index of the predicted class
        result = label_encoder.classes_[class_idx]  # Convert index to class label (yes/no)

        # Save the result to the database
        uploaded_image = UploadedImage(
            image=filename,
            result=result,
            area_mm2=area_mm2 if result == 'yes' else 0
        )
        uploaded_image.save()

        # Return the result in the response (pass image URLs to the template)
        return render(request, 'result.html', {
            'image_url': uploaded_image_url,
            'result': result,
            'area_mm2': area_mm2 if result == 'yes' else 0
        })

    return render(request, 'upload.html')
