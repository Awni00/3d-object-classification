import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np


from model import build_model

from azureml.core import Run

run = Run.get_context()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )

    args = parser.parse_args()

    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)

    IM_SIZE = (200, 200)
    seed = np.random.randint(0, int(1e6))
    rgb_train_path = f"{args.data_path}/train"
    rgb_dataset_train = image_dataset_from_directory(rgb_train_path, labels='inferred', label_mode='categorical',
                                                                    image_size=IM_SIZE, batch_size=32, shuffle=True,
                                                                    validation_split=0.05, subset='training', seed=seed)
    rgb_dataset_valid = image_dataset_from_directory(rgb_train_path, labels='inferred', label_mode='categorical',
                                                                    image_size=IM_SIZE, batch_size=32, shuffle=True,
                                                                    validation_split=0.05, subset='validation', seed=seed)

    rgb_test_path = f"{args.data_path}/test"
    rgb_dataset_test = image_dataset_from_directory(rgb_test_path, labels='inferred', label_mode='categorical',
                                                                    image_size=IM_SIZE, batch_size=32, shuffle=False)

    print('----------------')
    print('CLASS NAMES: ')
    print(rgb_dataset_train.class_names)
    print('----------------')


    # define model
    NUM_CLASSES = len(rgb_dataset_train.class_names)
    INPUT_SHAPE = [200, 200, 3]
    rgb_model = build_model(INPUT_SHAPE, NUM_CLASSES)

    rgb_model.summary() # print model summary

    rgb_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO: add callbacks to log into azure, save model checkpoints, early stopping
    rgb_model.fit(rgb_dataset_train, epochs=1, steps_per_epoch=20, validation_data=rgb_dataset_valid)

    loss, acc = rgb_model.evaluate(rgb_dataset_test)
    run.log('accuracy', np.float(acc))
    run.log('loss', np.float(loss))

    rgb_model.save_weights('outputs/rgb_model0.h5')



    print('Finished Training')