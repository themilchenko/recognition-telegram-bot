import tensorflow as tf
from PIL import Image

def prepare_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis


    return img_array
