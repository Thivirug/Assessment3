import cv2
import numpy as np
import tensorflow as tf

def calc_area(image: tf.Tensor, predicted_mask: np.ndarray) -> float:
    """
        Calculate the area of the segmented region in the mask.
    """
    # get amount of pixels in one squared centimeter

    image_gray = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2GRAY) # tf images are in RGB format
    
    ret, corners = cv2.findChessboardCorners(image_gray, (5,5), None)
    if ret:

        all_distances = []

        #calculate mean for higher accuracy
        position = 0
        for row in range(5):
            for col in range(4):
                dists = np.linalg.norm(corners[position] - corners[position+1])
                all_distances.append(dists)
                position += 1
            position += 1
        position = 0
        for col in range(5):
            for row in range(4):
                dists = np.linalg.norm(corners[position] - corners[position+1])
                all_distances.append(dists)
                position += 1
            position += 1
        mean_distance = np.mean(all_distances)
        print(f"Mean distance: {mean_distance}")
        area_in_pixels = mean_distance**2
        print(f"Area in pixels: {area_in_pixels}")

        # get area of mask in pixels
        pixel_count = cv2.countNonZero(predicted_mask)
        print(f"Pixel count: {pixel_count}")

        #get area in cm^2
        area = pixel_count / area_in_pixels

    else:
        raise ValueError("Reference grid not found in the image.")

    return area

