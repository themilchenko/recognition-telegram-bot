import numpy as np
from keras.preprocessing import image


def prepare_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, 0)
    # img = img.reshape(1, 256, 256, 3)

    return img / 255
