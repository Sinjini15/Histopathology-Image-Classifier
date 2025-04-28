# src/utils/path_utils.py

import os

ROOT_PATH_X = "/home/smitra16/.cache/kagglehub/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon/versions/9/pcam"
ROOT_PATH_Y = "/home/smitra16/.cache/kagglehub/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon/versions/9/Labels/Labels"

def generate_path():
    

    # Generate train, test and validation paths
    
    train_x_path = os.path.join(ROOT_PATH_X, 'training_split.h5' )
    train_y_path = os.path.join(ROOT_PATH_Y, 'camelyonpatch_level_2_split_train_y.h5')

    val_x_path = os.path.join(ROOT_PATH_X, 'validation_split.h5')
    val_y_path = os.path.join(ROOT_PATH_Y, 'camelyonpatch_level_2_split_valid_y.h5')

    test_x_path = os.path.join(ROOT_PATH_X, 'test_split.h5')
    test_y_path = os.path.join(ROOT_PATH_Y, 'camelyonpatch_level_2_split_test_y.h5')
    
    return train_x_path, train_y_path, test_x_path, test_y_path, val_x_path, val_y_path