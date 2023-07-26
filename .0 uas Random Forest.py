import glob
import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set the directory paths where your training and test images are located
training_directory = r"Training"  # Replace "path_to_training_directory" with the actual training directory path
test_directory = r"Test"  # Replace "path_to_test_directory" with the actual test directory path

# Define the file extensions or patterns for your training and test images
training_extension = "*.jpg"  
test_extension = "*.jpg"  

# Retrieve the file paths of the training images using glob
training_image_paths = glob.glob(os.path.join(training_directory, training_extension))

# Retrieve the file paths of the test images using glob
test_image_paths = glob.glob(os.path.join(test_directory, test_extension))

# Extract the label from the image file name
def extract_label(file_path):
    label = os.path.basename(file_path).split(".")[0].split("_")[0]
    return label

# Extract labels for training images
training_labels = [extract_label(path) for path in training_image_paths]

# Extract labels for test images
test_labels = [extract_label(path) for path in test_image_paths]

# Function to enhance and segment an image
def enhance_and_segment_image(image):
    # Define the parameters for adjusting contrast and brightness
    alpha = 1.28 # Contrast control 
    beta = -7.8  # Brightness control 

    # Apply contrast and brightness adjustment using cv2.addWeighted()
    enhanced_image = cv2.addWeighted(image, alpha, np.zeros(image.shape, dtype=image.dtype), 0, beta)

    # Convert the enhanced image from BGR to RGB for display
    img = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

    # Reshape image to 2D
    twoDimage = img.reshape((-1, 3))

    # Convert image data type to float32
    twoDimage = np.float32(twoDimage)

    # Set criteria for K-Means algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6  # Number of color clusters
    attempts = 10  # Number of attempts

    # Apply K-Means algorithm to the enhanced image
    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()] 
    segmented_image = res.reshape((img.shape))

    return segmented_image

# Function to extract color features from an image
def extract_color_features(image):
    # Convert the segmented image from RGB to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract color features from the HSV image (e.g., mean hue, mean saturation, mean value)
    mean_hue = np.mean(hsv_image[:, :, 0])
    mean_saturation = np.mean(hsv_image[:, :, 1])
    mean_value = np.mean(hsv_image[:, :, 2])

    return [mean_hue, mean_saturation, mean_value]

# Create lists to store the segmented images and extracted features
training_segmented_images = []
training_features = []
test_segmented_images = []
test_features = []

# Process the training images
for image_path in training_image_paths:
    # Read the image
    image = cv2.imread(image_path)

    # Enhance and segment the image
    segmented_image = enhance_and_segment_image(image)

    # Append the segmented image to the list
    training_segmented_images.append(segmented_image)

    # Extract color features from the segmented image
    features = extract_color_features(segmented_image)

    # Append the features to the list
    training_features.append(features)

# Process the test images
for image_path in test_image_paths:
    # Read the image
    image = cv2.imread(image_path)

    # Enhance and segment the image
    segmented_image = enhance_and_segment_image(image)

    # Append the segmented image to the list
    test_segmented_images.append(segmented_image)

    # Extract color features from the segmented image
    features = extract_color_features(segmented_image)

    # Append the features to the list
    test_features.append(features)

# Convert the lists to numpy arrays
training_features = np.array(training_features)
training_labels = np.array(training_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

print (test_labels) 
print (test_features)

# Create and train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100)
model.fit(training_features, training_labels)

# Use the trained model to predict labels for the test features
predicted_labels = model.predict(test_features)

# Print the actual label and predicted label for each test image
for i in range(len(test_image_paths)):
    image_path = test_image_paths[i]
    image = cv2.imread(image_path)
    actual_label = test_labels[i]
    predicted_label = predicted_labels[i]

    print()
    print("Actual Label   :", actual_label)
    print("Predicted Label:", predicted_label)
    

    # Display the test image
    cv2.imshow("Test Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Evaluate the model's performance
accuracy = 100*(np.mean(predicted_labels == test_labels))
print("Accuracy:", accuracy, '%')
