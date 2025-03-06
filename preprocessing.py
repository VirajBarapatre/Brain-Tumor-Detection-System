import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Define dataset path
dataset_path = 'C:\\Users\\viraj\\brain_tumor\\dataset'
output_path = 'Brain_Tumor_Data_Preprocessed'
img_size = (150, 150)

# Create output folders
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(f'{output_path}/train', exist_ok=True)
os.makedirs(f'{output_path}/test', exist_ok=True)

# Label Mapping
def get_label(folder_name):
    if folder_name in ['glioma', 'meningioma', 'pituitary']:
        return 1  # Tumor
    else:
        return 0  # No Tumor

# Load Dataset
X, Y = [], []

for folder in ['Training', 'Testing']:
    folder_path = os.path.join(dataset_path, folder)
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            label = get_label(subfolder)
            for img in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, img)
                try:
                    image = cv2.imread(img_path)
                    if image is not None:
                        image = cv2.resize(image, img_size)
                        X.append(image)
                        Y.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

# Convert to NumPy arrays
X = np.array(X) / 255.0  # Normalize Images
Y = np.array(Y)

# Split Dataset into Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Save Preprocessed Data
def save_images(X, Y, folder):
    for i, img in enumerate(X):
        label_folder = 'tumor' if Y[i] == 1 else 'no_tumor'
        folder_path = f'{output_path}/{folder}/{label_folder}'
        os.makedirs(folder_path, exist_ok=True)
        img_path = os.path.join(folder_path, f'{i}.jpg')
        cv2.imwrite(img_path, img * 255)

save_images(X_train, Y_train, 'train')
save_images(X_test, Y_test, 'test')

print("✅ Data Preprocessing Completed Successfully!")
