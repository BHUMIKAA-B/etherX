from PIL import Image, ExifTags
import cv2
import numpy as np


def load_image(path):
    image = Image.open(path)

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)

            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)

    except:
        pass

    # Convert PIL → OpenCV
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

