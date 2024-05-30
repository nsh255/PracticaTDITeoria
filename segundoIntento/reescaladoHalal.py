import os
from PIL import Image

# Define the directory containing the JPG images
image_dir = "segundoIntento/data/n"

# Get a list of all JPG files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Loop through each image file
for i, image_file in enumerate(image_files):
    # Open the image
    image = Image.open(os.path.join(image_dir, image_file))

    # Resize the image to the desired dimensions
    resized_image = image.resize((24, 24))

    # Save the resized image with a new filename
    resized_image.save(f"image_{i+1}.jpg")
