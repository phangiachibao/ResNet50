import os
import shutil
import pandas as pd


def __data_into_train_test(
        dataset_path,
        training_set_path,
        test_set_path,
        classes
):
    for c in classes:
        # Class path
        dataset_class = dataset_path + '/' + c
        training_class = training_set_path + '/' + c
        test_class = test_set_path + '/' + c
        # Read all image of recent class
        df = pd.DataFrame(os.listdir(dataset_class))
        # Shuffle data
        df = df.sample(frac=1).values
        # Get 80% to training_set, 20% to test_set(can change here)
        train_len = int(len(df) * 80 / 100)
        for i in range(len(df)):
            if i <= train_len:
                shutil.copy(dataset_class + '/' + df[i, 0],
                            training_class + '/' + c + '_' + df[i, 0])
            else:
                shutil.copy(dataset_class + '/' + df[i, 0],
                            test_class + '/' + c + '_' + df[i, 0])


# Create training_set, test_set
# 1. Create folders training_set, test_set
# 2. Create class folders into training_set, test_set
# 3. Put images from original folder into training_set, test_set
# then, rename all images with
# prefix is those class
def create_train_test_set(dataset_path):
    # Part 1
    classes = os.listdir(dataset_path)
    # Path training_set, test_set
    training_set_path = 'dataset/training_set'
    test_set_path = 'dataset/test_set'
    # Create training_set, test_set folders
    os.mkdir(training_set_path)
    os.mkdir(test_set_path)
    # Part 2
    # Create class folders into training_set, test_set
    for c in classes:
        os.mkdir(training_set_path + '/' + c)
        os.mkdir(test_set_path + '/' + c)
    # Part 3
    __data_into_train_test(
        dataset_path,
        training_set_path,
        test_set_path,
        classes
    )


create_train_test_set('dataset/Multi-class Weather Dataset')
