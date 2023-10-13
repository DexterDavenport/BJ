'''
To use the virtual environment run the correct code for your operating system
in the terminal before running the code...

macOS:
source venv/bin/activate

Windows:
.\venv\Scripts\activate
'''


import cv2  # Importing OpenCV
import pytesseract  # Importing Tesseract
import os
import numpy as np  # Importing numpy for mask calculations

# Specify the folder containing images
image_folder = "images/Top20"

# List all files in the image folder
image_files = [
    f
    for f in os.listdir(image_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
]


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def get_mask_from_boxes(boxes, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Initialize with zeros
    for box in boxes.splitlines():
        box_values = box.split()
        if len(box_values) >= 6 and all(is_int(val) for val in box_values[1:5]):
            x1, y1, x2, y2 = map(int, box_values[1:5])
            mask[y1:y2, x1:x2] = 1  # Mark the region as text
    return mask


failed_images = []  # List to keep track of failed images

print("=== Successfully Processed Images ===")

for image_file in image_files:
    # Load the image
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        failed_images.append(image_path)  # Store the failed image path
        continue  # Skip this iteration and move to the next image

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply OCR to detect text with tuned configuration
    recognized_text = pytesseract.image_to_string(gray_image, config="--psm 6")

    # Get the bounding boxes for the detected text with the same configuration
    boxes = pytesseract.image_to_boxes(gray_image, config="--psm 6")

    # Calculate the text area using a mask
    text_mask = get_mask_from_boxes(boxes, gray_image.shape)
    text_area = np.sum(text_mask)

    # Calculate the total image area
    total_area = gray_image.shape[0] * gray_image.shape[1]

    # Calculate the percentage of the image containing text
    percentage_text_area = (text_area / total_area) * 100

    print(f"Percentage of text in {image_file}: {percentage_text_area:.2f}%")

print("\n=== Failed Images ===")
for failed_image in failed_images:
    print(f"Could not read image: {failed_image}")
