import os
import time
import argparse
import logging

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from model import build_model, AzureLogCallback
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

    print('loading rgbd train and test datasets')

    rgbd_dataset = utils.create_rgbd_dataset(train_data_path)
    test_dataset = utils.create_rgbd_dataset(test_data_path)

    print(f'done! took {time.time()-start_time:.2f} seconds')

    print('----------------')
    print('CLASS NAMES: ')
    print(rgbd_dataset.class_names)
    print('----------------')

    num_train_examples = len(rgbd_dataset)
    print('num train examples: ', num_train_examples)
    run.log('num train examples', num_train_examples)

    BUFFER_SIZE = 10000
    # BATCH_SIZE_PER_REPLICA = 64
    # BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    BATCH_SIZE = 16 # NOTE: temporarily decrease batch size to test OOM problem

    train_dataset = rgbd_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # build model
    IM_SIZE = (200, 200)
    NUM_CLASSES = len(rgbd_dataset.class_names)
    print('num classes: ', NUM_CLASSES)

    # Build model
    print('building model...')
    start_time = time.time()
    weights_loaded = False
    # with strategy.scope():
    rgbd_model = build_model(IM_SIZE, NUM_CLASSES)
    try:
        weights_path = f'{args.model_weights_path}/{args.model_weights_name}'
        print('loading weights from ', weights_path)
        rgbd_model.load_weights(weights_path)
        weights_loaded = True
    except Exception as e:
        print(e)
        print('no weights were loaded to the model.')

    print(f'done! took {time.time()-start_time:.2f} seconds')
    rgbd_model.summary()  # print model summary

    # log trainable parameters of model
    trainable_count = np.sum([K.count_params(w) for w in rgbd_model.trainable_weights])
    run.log('trainable parameters', trainable_count)


    # prep model for training
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_metric = tf.keras.metrics.CategoricalAccuracy()
    test_metric = tf.keras.metrics.CategoricalAccuracy()

    # rgbd_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    # only evaluate against test dataset before training if weights were loaded
    if weights_loaded:
        print("testing against test dataset before training")
        loss, acc = rgbd_model.evaluate(test_dataset)
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
    # epoch_start = 0
    # n_epochs = 4
    # rgbd_model.fit(train_dataset, # validation_data=test_dataset,
    #                 epochs=epoch_start + n_epochs, initial_epoch=epoch_start,
    #                 callbacks=[model_checkpoint_cb, azure_log_cb])

    os.mkdir('outputs/checkpoints') # make checkpoints directory

    # training loop
    epochs = 3

    for epoch in range(1, epochs + 1):
        # training batches
        losses_train = train_utils.train_epoch(rgbd_model, train_dataset, epoch, optimizer, loss, train_metric, max_steps=None, logger=logger)

        train_acc = np.float(train_metric.result()) # get train acc from training batches
        train_metric.reset_states() # reset train acc for next epoch

        # calculate validation metrics
        losses_val = train_utils.validate_epoch(rgbd_model, test_dataset, epoch, loss, test_metric, max_steps=None, logger=logger)

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
        rgbd_model.save_weights(f'outputs/checkpoints/epoch{epoch}_weights.h5')


    print('done training!')

    # evaluate on test dataset
    # print('evaluating on test dataset...')
    # loss, acc = rgbd_model.evaluate(test_dataset)
    # run.log('test accuracy', np.float(acc))
    # run.log('test loss', np.float(loss))

    print('saving model weights...')
    # rgbd_model.save_weights('outputs/rgbd_model_weights.h5')
    rgbd_model.save('outputs/rgbd_model')

    print('All done!')
