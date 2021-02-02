import os
import time
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np


from rgb_model import build_model, AzureLogCallback
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
    parser.add_argument(
        '--model_weights_path',
        type=str,
        help='path to model weights to load before training')
    parser.add_argument(
    '--model_weights_name',
    type=str,
    help='filename of model weights')

    args = parser.parse_args()

    print("===== INPUTS =====")
    print("DATA PATH: " + args.data_path)
    if args.model_weights_path and args.model_weights_name:
        print("MODEL WEIGHTS PATH: " + args.model_weights_path)
        print("MODEL WEIGHTS NAME: " + args.model_weights_name)

        print('printing lsdir in weights path')
        print(os.listdir(args.model_weights_path))

    # LOAD DATASET
    train_data_path = f"{args.data_path}/train"
    test_data_path = f"{args.data_path}/test"

    start_time = time.time()

    print('loading rgb train and test datasets')

    rgb_dataset = utils.create_rgb_dataset(train_data_path)
    test_dataset = utils.create_rgb_dataset(test_data_path)

    print(f'done! took {time.time()-start_time:.2f} seconds')

    print('----------------')
    print('CLASS NAMES: ')
    print(rgb_dataset.class_names)
    print('----------------')

    # define training strategy
    strategy = tf.distribute.MirroredStrategy()

    # the number of devices
    print(f'number of devices: {strategy.num_replicas_in_sync}')
    run.log('number of devices', strategy.num_replicas_in_sync)

    num_train_examples = len(rgb_dataset)
    print('num train examples: ', num_train_examples)
    run.log('num train examples', num_train_examples)

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    train_dataset = rgb_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # build model
    IM_SIZE = (200, 200)
    NUM_CLASSES = len(rgb_dataset.class_names)
    print('num classes: ', NUM_CLASSES)

    # with strategy.scope(): # TODO
    print('building model...')
    start_time = time.time()
    with strategy.scope():
        rgb_model = build_model(IM_SIZE, NUM_CLASSES)
        try:
            weights_path = f'{args.model_weights_path}/{args.model_weights_name}'
            print('loading weights from ', weights_path)
            rgb_model.load_weights(weights_path)
        except Exception as e:
            print(e)
            print('no weights were loaded to the model.')

    print(f'done! took {time.time()-start_time:.2f} seconds')
    rgb_model.summary()  # print model summary

    # compile model
    loss = tf.keras.losses.CategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam()

    rgb_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    # TEMP: test that weights were loaded using test dataset
    print("testing against test dataset before training")
    loss, acc = rgb_model.evaluate(test_dataset)
    run.log('pre-training test accuracy', np.float(acc))
    run.log('pre-training test loss', np.float(loss))
    print('done and logged!')


    model_filepath = 'outputs/model0'
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        model_filepath, monitor='loss', save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')

    azure_log_cb = AzureLogCallback(run, metrics_to_log=['loss', 'accuracy'])

    # TODO: add callbacks for: early stopping, ...
    # TODO: dataset prefetch
    # TODO: use distributed training strategy for multiple gpus
    # TODO: image augmentation
    print('beginning training...')
    epoch_start = 0
    n_epochs = 6
    rgb_model.fit(train_dataset, epochs=epoch_start + n_epochs, initial_epoch=epoch_start,
                    callbacks=[model_checkpoint_cb, azure_log_cb])  # validation_data=test_dataset
    print('done training!')

    # evaluate on test dataset
    print('evaluating on test dataset...')
    loss, acc = rgb_model.evaluate(test_dataset)
    run.log('test accuracy', np.float(acc))
    run.log('test loss', np.float(loss))

    print('saving model weights...')
    rgb_model.save_weights('outputs/rgb_model0_weights.h5')
    rgb_model.save('outputs/rgb_model0')

    print('All done!')
