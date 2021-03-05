import os
import time
import argparse
import logging

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from rgb_hha_model import build_hha_feature_extractor, build_rgb_hha_model
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
        '--hha_weights_path',
        type=str,
        help='Path to the hha model\'s weights'
    )
    parser.add_argument(
        '--model_weights_path',
        type=str,
        help='path to model weights to load before training')


    args = parser.parse_args()

    print("===== INPUTS =====")
    print("DATA PATH: " + args.data_path)
    if (args.model_weights_path is not None):
        print("MODEL WEIGHTS PATH: " + args.model_weights_path)

    # set up logging
    logging.basicConfig(filename='outputs/train.log', level=logging.ERROR,
                        format='%(levelname)s %(asctime)s - %(message)s')
    logger = logging.getLogger()

    # LOAD DATASET
    train_data_path = f"{args.data_path}/train"
    test_data_path = f"{args.data_path}/test"

    start_time = time.time()

    print('loading rgb-hha train and test datasets')

    rgb_hha_dataset = utils.create_rgb_hha_dataset(train_data_path)
    test_dataset = utils.create_rgb_hha_dataset(test_data_path)

    print(f'done! took {time.time()-start_time:.2f} seconds')

    print('----------------')
    print('CLASS NAMES: ')
    print(rgb_hha_dataset.class_names)
    print('----------------')

    num_train_examples = len(rgb_hha_dataset)
    print('num train examples: ', num_train_examples)
    run.log('num train examples', num_train_examples)

    BUFFER_SIZE = 10000
    # BATCH_SIZE_PER_REPLICA = 64
    # BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    BATCH_SIZE = 16 # NOTE: temporarily decrease batch size to test OOM problem

    train_dataset = rgb_hha_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # build model
    IM_SIZE = (200, 200)
    NUM_CLASSES = len(rgb_hha_dataset.class_names)
    print('num classes: ', NUM_CLASSES)

    # Build model
    print('building model...')
    start_time = time.time()

    # first, build pre-trained hha feature extractor
    hha_feat_extractor = build_hha_feature_extractor(
        weights=args.hha_weights_path, trainable=True)
    hha_feat_extractor.summary()

    # now build rgb-hha model
    rgb_hha_model = build_rgb_hha_model(input_shape=(200, 200), num_classes=51,
        hha_feat_vec_embedding=hha_feat_extractor, rgb_feat_vec_embedding=None,
        model_name='rgb-hha_model')
    rgb_hha_model.summary()

    weights_loaded = False
    try:
        weights_path = f'{args.model_weights_path}'
        print('loading weights from ', weights_path)
        rgb_hha_model.load_weights(weights_path)
        weights_loaded = True
    except Exception as e:
        print(e)
        print('no weights were loaded to the model.')

    print(f'done! took {time.time()-start_time:.2f} seconds')
    rgb_hha_model.summary()  # print model summary

    # log trainable parameters of model
    trainable_count = np.sum([K.count_params(w) for w in rgb_hha_model.trainable_weights])
    run.log('trainable parameters', trainable_count)


    # prep model for training
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_metric = tf.keras.metrics.CategoricalAccuracy()
    test_metric = tf.keras.metrics.CategoricalAccuracy()

    # TRAINING
    print('beginning training...')

    os.mkdir('outputs/checkpoints') # make checkpoints directory

    # training loop
    epochs = 8

    for epoch in range(1, epochs + 1):
        # training batches
        losses_train = train_utils.train_epoch(rgb_hha_model, train_dataset, epoch, optimizer, loss, train_metric, max_steps=None, logger=logger)

        train_acc = np.float(train_metric.result()) # get train acc from training batches
        train_metric.reset_states() # reset train acc for next epoch

        # calculate validation metrics
        losses_val = train_utils.validate_epoch(rgb_hha_model, test_dataset, epoch, loss, test_metric, max_steps=None, logger=logger)

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
        rgb_hha_model.save_weights(f'outputs/checkpoints/epoch{epoch}_weights.h5')


    print('done training!')


    print('saving model weights...')
    rgb_hha_model.save('outputs/rgb_hha_model')

    print('All done!')
