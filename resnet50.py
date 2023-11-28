from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, ZeroPadding2D


def identity_block(x, filters):
    """
    Description:
        Build identity block
        Layers used: Conv2D, BatchNormalization, Activation, Add
    """
    # filters for Layer 1, 2, 3 behind
    f1, f2, f3 = filters
    # Retrieve x by x_shortcut
    x_shortcut = x
    # All strides of 3 Layers equal 1
    # layer 1
    x = Conv2D(filters=f1, kernel_size=1)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    # Layer 2
    x = Conv2D(filters=f2, kernel_size=3, padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    # Layer 3: F(x) - Don't have relu here
    x = Conv2D(filters=f3, kernel_size=1)(x)
    x = BatchNormalization(axis=3)(x)
    # Final Layer: F(x) + x
    x = Add()([x, x_shortcut])
    # ReLu function
    x = Activation("relu")(x)
    return x


def convolutional_block(x, filters, s=1):
    """
    Description:
        Build convolutional block
        Layers used: Conv2D, BatchNormalization, Activation, Add
    """
    f1, f2, f3 = filters
    # Retrieve x by x_shortcut
    x_shortcut = x
    # Layer 1
    x = Conv2D(filters=f1, kernel_size=1, strides=s)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filters=f2, kernel_size=3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Layer 3 : F(x) - Don't have relu here
    x = Conv2D(filters=f3, kernel_size=1)(x)
    x = BatchNormalization(axis=3)(x)
    # Shortcut Layer - same Layer 3 but this is in x_shortcut
    x_shortcut = Conv2D(filters=f3, kernel_size=1, strides=s)(x_shortcut)
    x_shortcut = BatchNormalization(axis=3)(x_shortcut)
    # Final Layer: F(x) + x
    x = Add()([x, x_shortcut])
    # ReLu function
    x = Activation('relu')(x)
    return x


class ResNet50:
    """
    Description:
        A class that will create an instance of Residual Network Model
    """
    def __init__(self):
        self.img_size = None
        self.model = None
        self.classes = None

    def preprocess_dataset(self, batch_size):
        """
        Description:
            A method that will preprocess images data to avoid the over fitting.
        """
        # preprocessing and make training set
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        training_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )
        # preprocessing and make test set
        test_datagen = ImageDataGenerator(
            rescale=1./255
        )
        test_set = test_datagen.flow_from_directory(
            'dataset/test_set',
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )
        # Get classes
        self.classes = training_set.class_indices
        print(self.classes)
        return training_set, test_set

    def create_model(self, input_shape):
        """
        Description:
            Build ResNet50 model
            Layers used: Input, ZeroPadding2D, Conv2D, Activation,
            AveragePooling2D, Flatten, Dense
        """
        # Tensor input
        x_input = Input(input_shape)  # Results (None, input_shape)
        x = ZeroPadding2D(padding=3)(x_input)
        # Stage 1
        x = Conv2D(filters=64, kernel_size=7, strides=2)(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)
        # Stage 2 - 3 times
        x = convolutional_block(x, [64, 64, 256])
        x = identity_block(x, [64, 64, 256])
        x = identity_block(x, [64, 64, 256])
        # Stage 3 - 4 times
        x = convolutional_block(x, [128, 128, 512], 2)
        x = identity_block(x, [128, 128, 512])
        x = identity_block(x, [128, 128, 512])
        x = identity_block(x, [128, 128, 512])
        # Stage 4 - 6 times
        x = convolutional_block(x, [256, 256, 1024], 2)
        x = identity_block(x, [256, 256, 1024])
        x = identity_block(x, [256, 256, 1024])
        x = identity_block(x, [256, 256, 1024])
        x = identity_block(x, [256, 256, 1024])
        x = identity_block(x, [256, 256, 1024])
        # Stage 5 - 3 times
        x = convolutional_block(x, [512, 512, 2048], 2)
        x = identity_block(x, [512, 512, 2048])
        x = identity_block(x, [512, 512, 2048])
        # Average Pooling
        x = AveragePooling2D(pool_size=2, padding='same')(x)
        # Flatten x
        x = Flatten()(x)
        # Build Neural Network
        x = Dense(units=len(self.classes), activation='softmax')(x)
        # Build model
        self.model = Model(inputs=x_input, outputs=x, name='resnet50_model')

    def training_resnet50(self, img_size, batch_size, epochs, model_name, path):
        self.img_size = img_size
        dataset = self.preprocess_dataset(batch_size)
        self.create_model(input_shape=(self.img_size, self.img_size, 3))
        # Model training
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(x=dataset[0], validation_data=dataset[1], epochs=epochs)
        self.save_model(model_name, path)

    def save_model(self, model_name, path):
        self.model.save(path + model_name)
