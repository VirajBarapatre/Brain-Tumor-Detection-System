import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk

# Load the trained model
model = load_model('brain_tumor_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict the image
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor Detected"

# Function to upload and display image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        result = predict_image(file_path)
        result_label.config(text=f"Result: {result}")

# GUI Window
root = tk.Tk()
root.title("Brain Tumor Detection System")
root.geometry("500x600")
root.config(bg="lightblue")

Label(root, text="Brain Tumor Detection System", font=("Arial", 20, "bold"), bg="lightblue").pack(pady=20)
image_label = Label(root)
image_label.pack(pady=20)

Button(root, text="Upload MRI Image", command=upload_image, font=("Arial", 14), bg="green", fg="white").pack(pady=20)
result_label = Label(root, text="", font=("Arial", 16, "bold"), bg="lightblue")
result_label.pack(pady=20)

root.mainloop()
