import os
import time
import argparse
import logging

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from hha_model import build_model
import utils
import train_utils

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
    if (args.model_weights_path is not None) and (args.model_weights_name is not None):
        print("MODEL WEIGHTS PATH: " + args.model_weights_path)
        print('MODEL WEIGHTS NAME: ' + args.model_weights_name)

    # set up logging
    logging.basicConfig(filename='outputs/train.log', level=logging.ERROR,
                        format='%(levelname)s %(asctime)s - %(message)s')
    logger = logging.getLogger()

    # LOAD DATASET
    train_data_path = f"{args.data_path}/train"
    test_data_path = f"{args.data_path}/test"

    start_time = time.time()

    print('loading hha train and test datasets')

    hha_dataset = utils.create_hha_dataset(train_data_path)
    test_dataset = utils.create_hha_dataset(test_data_path)

    print(f'done! took {time.time()-start_time:.2f} seconds')

    print('----------------')
    print('CLASS NAMES: ')
    print(hha_dataset.class_names)
    print('----------------')

    num_train_examples = len(hha_dataset)
    print('num train examples: ', num_train_examples)
    run.log('num train examples', num_train_examples)

    BUFFER_SIZE = 10000
    BATCH_SIZE = 16 # NOTE: decreased batch size to fix OOM problem

    train_dataset = hha_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # build model
    IM_SIZE = (200, 200)
    NUM_CLASSES = len(hha_dataset.class_names)
    print('num classes: ', NUM_CLASSES)

    # Build model
    print('building model...')
    start_time = time.time()
    weights_loaded = False
    # with strategy.scope():
    hha_model = build_model(IM_SIZE, NUM_CLASSES)
    try:
        weights_path = f'{args.model_weights_path}/{args.model_weights_name}'
        print('loading weights from ', weights_path)
        hha_model.load_weights(weights_path)
        weights_loaded = True
    except Exception as e:
        print(e)
        print('no weights were loaded to the model.')

    print(f'done! took {time.time()-start_time:.2f} seconds')
    hha_model.summary()  # print model summary

    # log trainable parameters of model
    trainable_count = np.sum([K.count_params(w) for w in hha_model.trainable_weights])
    run.log('trainable parameters', trainable_count)


    # prep model for training
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_metric = tf.keras.metrics.CategoricalAccuracy()
    test_metric = tf.keras.metrics.CategoricalAccuracy()

    # only evaluate against test dataset before training if weights were loaded
    if weights_loaded:
        print("testing against test dataset before training")
        loss, acc = hha_model.evaluate(test_dataset)
        run.log('pre-training test accuracy', np.float(acc))
        run.log('pre-training test loss', np.float(loss))
        print('done and logged!')

    # # define callbacks
    # model_filepath = 'outputs/model'
    # model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    #     model_filepath, monitor='loss', save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')

    # azure_log_cb = AzureLogCallback(run, metrics_to_log=['loss', 'accuracy', 'val_loss', 'val_accuracy'])


    # TODO: add callbacks for: early stopping, ...
    # TODO: image augmentation
    print('beginning training...')


    os.mkdir('outputs/checkpoints') # make checkpoints directory

    # training loop
    epochs = 3

    for epoch in range(1, epochs + 1):
        # training batches
        losses_train = train_utils.train_epoch(hha_model, train_dataset, epoch, optimizer, loss, train_metric, max_steps=None, logger=logger)

        train_acc = np.float(train_metric.result()) # get train acc from training batches
        train_metric.reset_states() # reset train acc for next epoch

        # calculate validation metrics
        losses_val = train_utils.validate_epoch(hha_model, test_dataset, epoch, loss, test_metric, max_steps=None, logger=logger)

        val_acc = np.float(test_metric.result()) # get test acc from training batches
        test_metric.reset_states() # reset test acc for next epoch

        losses_train_mean = np.mean(losses_train)
        losses_val_mean = np.mean(losses_val)

        print(f'epoch: [{epoch}/{epochs}]; acc: {train_acc}; loss: {losses_train_mean}; val_acc: {val_acc}; val_loss: {losses_val_mean}')

        # log to azure run
        metric_logs = {'acc': train_acc, 'loss': losses_train_mean, 'val_acc': val_acc, 'val_loss': losses_val_mean}
        for metric, value in metric_logs.items():
            run.log(metric, value)

        print('saving model...')
        hha_model.save_weights(f'outputs/checkpoints/epoch{epoch}_weights.h5')


    print('done training!')


    print('saving model weights...')
    hha_model.save('outputs/hha_model')

    print('All done!')
