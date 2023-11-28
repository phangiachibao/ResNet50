from predict import Predict
# from predict import show_img


def main():
    # Initialize instance of Predict
    pred = Predict('trained_models/Resnet50.keras')
    # Give an image path here
    image_path = 'test_image.jpg'
    # Show image will be predicted
    # show_img(image_path)
    # Predict an image
    pred.predict_image(image_path)


if __name__ == '__main__':
    main()
