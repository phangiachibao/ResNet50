from PIL import Image
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt


def show_img(image_path):
    """
    Description:
        A function that will show image from a given path

    Parameter:
    ----------
    image_path
        String. A path of an image.
    """
    img = np.asarray(Image.open(image_path))
    plt.imshow(img)
    plt.show()


class Predict:
    """
    Description:
        A class that will predict a given image.
    """
    def __init__(self, link_model):
        self.model = load_model(link_model)

    def predict_image(self, image_path):
        """
        Description:
            A method that will show predicted probabilities of classes when
            passing a path of an image

        Parameter
        ----------
        image_path
            String. A path of an image want to predict.
        """
        # Load an image
        test_image = load_img(image_path, target_size=(224, 224))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        # Scale an image to have a same size with training images
        result = self.model.predict(test_image/255.0)
        # Define a dictionary of classes
        classes = {'Cloudy': 0,
                   'Rain': 1,
                   'Shine': 2,
                   'Sunrise': 3}
        # Print predicted probabilities each classes
        for class_name, v in classes.items():
            print(f"{class_name}: {result[0, v] * 100:.2f}%")
