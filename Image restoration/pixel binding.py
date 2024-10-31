from PIL import Image
import numpy as np
import tifffile as tiff

def bind_pixels(image_path):
    # Load the original image
    original_image = tiff.imread(image_path)
    print(original_image.shape)
    # Convert the image to a numpy array
    original_image = np.array(original_image)
    

    # Get the original image size
    depth, height, width = original_image.shape

    # Calculate the new size
    new_width = 256
    new_height = 256

    # Create a new image with the new size
    new_image = np.zeros((depth, new_height, new_width))

    # Iterate over the new image pixels
    for z in range(depth):
        for y in range(new_height):
            for x in range(new_width):
                # Calculate the corresponding pixel coordinates in the original image
                original_x = x * (width // new_width)
                original_y = y * (height // new_height)

                # Get the pixels from the original image
                pixels = original_image[z, original_y:original_y + (height // new_height), original_x:original_x + (width // new_width)]

                # Calculate the average pixel value
                average_pixel = np.mean(pixels)

                # Set the new pixel value in the new image
                new_image[z, y, x] = average_pixel

    return new_image

# Example usage
original_image_path = 'Z:/Users/yj368/Autofocus_repeat/Autofocus_repeat/ER_Yutong/repeat8.1/Stack1.tif'
resized_image = bind_pixels(original_image_path)
resized_image = np.resize(resized_image, (resized_image.shape[0], 256, 256))
resized_image = resized_image.astype(np.uint16)
tiff.imsave('Z:/Users/yj368/Autofocus_repeat/Autofocus_repeat/ER_Yutong/repeat8.1/resized_image.tif', resized_image)
# resized_image.save('Z:/Users/yj368/Autofocus_repeat/Autofocus_repeat/ER_Yutong/Stack_-10_to_10_1umstep/resized_image.tif')
