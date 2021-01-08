import os
import time
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np


from model import build_model, AzureLogCallback
import utils

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

    # LOAD DATASET
    train_data_path = f"{args.data_path}/train"
    test_data_path = f"{args.data_path}/test"

    start_time = time.time()

    print('loading rgbd train and test datasets')

    rgbd_dataset = utils.create_rgbd_dataset(train_data_path)
    test_dataset = utils.create_rgbd_dataset(test_data_path)

    print(f'done! took {time.time()-start_time:.2f} seconds')

    print('----------------')
    print('CLASS NAMES: ')
    print(rgbd_dataset.class_names)
    print('----------------')

    # define training strategy (NOTE: not used yet in this script)
    strategy = tf.distribute.MirroredStrategy()
    # the number of devices
    print(f'number of devices: {strategy.num_replicas_in_sync}')
    run.log('number of devices', strategy.num_replicas_in_sync)
    print('gpu device name: ', tf.test.gpu_device_name())
    run.log('gpu device name', tf.test.gpu_device_name())

    num_train_examples = len(rgbd_dataset)
    print('num train examples: ', num_train_examples)

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    train_dataset = rgbd_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # build model
    IM_SIZE = (200, 200)
    NUM_CLASSES = len(rgbd_dataset.class_names)
    print('num classes: ', NUM_CLASSES)

    # with strategy.scope(): # TODO
    print('building model...')
    start_time = time.time()
    rgbd_model = build_model(IM_SIZE, NUM_CLASSES)
    print(f'done! took {time.time()-start_time:.2f} seconds')
    rgbd_model.summary()  # print model summary

    # compile model normally
    loss = tf.keras.losses.CategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam()

    rgbd_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    model_filepath = 'output/model0'
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        model_filepath, monitor='loss', save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')

    azure_log_cb = AzureLogCallback(run, metrics_to_log=['loss', 'accuracy'])

    # TODO: add callbacks for: early stopping, ...
    # TODO: dataset prefetch
    # TODO: use distributed training strategy for multiple gpus
    # TODO: image augmentation
    EPOCHS = 1
    rgbd_model.fit(train_dataset, epochs=EPOCHS, callbacks=[
                   model_checkpoint_cb, azure_log_cb])  # validation_data=test_dataset, #NOTE: steps_per_epoch TEMP

    # evaluate on test dataset
    loss, acc = rgbd_model.evaluate(test_dataset)
    run.log('test accuracy', np.float(acc))
    run.log('test loss', np.float(loss))

    rgbd_model.save_weights('outputs/rgbd_model0.h5')

    print('Finished Training')
