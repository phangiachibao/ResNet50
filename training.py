from resnet50 import ResNet50
import os


# Fit environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = ResNet50()
model.training_resnet50(img_size=224,
                        batch_size=32,
                        epochs=10,
                        model_name="Resnet50.keras",
                        path="trained_models/")
