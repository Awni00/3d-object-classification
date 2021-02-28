import os
import time
import argparse
import logging
import importlib

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

import rgb_model, rgbd_model, depth_model, hha_model, rgb_hha_model
import utils
import train_utils
import eval_utils

from azureml.core import Run

run = Run.get_context()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the dataset'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        help='type of model (rgb, rgbd, hha, etc...)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        help='filename of model to be loaded'
    )
    parser.add_argument(
        '--model_weights_name',
        type=str,
        help='filename of model weights')
    # parser.add_argument(
    #     '--model_weights_path',
    #     type=str,
    #     help='path to model weights to load before training')


    args = parser.parse_args()
    run.log('model type', args.model_type)

    print("===== INPUTS =====")
    print("DATA PATH: " + args.data_path)
    # if (args.model_weights_path is not None) and (args.model_weights_name is not None):
    #     print("MODEL WEIGHTS PATH: " + args.model_weights_path)
    #     print('MODEL WEIGHTS NAME: ' + args.model_weights_name)

    # LOAD DATASET
    train_data_path = f"{args.data_path}/train"
    test_data_path = f"{args.data_path}/test"

    start_time = time.time()

    print('loading datasets')

    # get dataset loader depending on model_type
    if args.model_type == 'rgb':
        dataset_loader = utils.create_rgb_dataset
    elif args.model_type == 'depth':
        dataset_loader = utils.create_depth_dataset
    elif args.model_type == 'rgb-depth':
        dataset_loader = utils.create_rgbd_dataset
    elif args.model_type == 'hha':
        dataset_loader = utils.create_hha_dataset
    elif args.model_type == 'rgb-hha':
        dataset_loader = utils.create_rgb_hha_dataset
    else:
        raise ValueError(f'model_type argument {args.model_type} is not recognized.')

    # dataset = dataset_loader(train_data_path)
    test_dataset = dataset_loader(test_data_path)

    print(f'done! took {time.time()-start_time:.2f} seconds')

    print('----------------')
    print('CLASS NAMES: ')
    print(test_dataset.class_names)
    print('----------------')

    # num_train_examples = len(dataset)
    # print('num train examples: ', num_train_examples)
    # run.log('num train examples', num_train_examples)

    BUFFER_SIZE = 10000
    # BATCH_SIZE_PER_REPLICA = 64
    # BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    BATCH_SIZE = 512

    # train_dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # build model
    IM_SIZE = (200, 200)
    NUM_CLASSES = 51
    print('num classes: ', NUM_CLASSES)

    # Build model
    print('building model...')
    start_time = time.time()

    # get model builder from model file
    model_builder = importlib.import_module(f'models_archive.{args.model_name}').build_model
    model = model_builder(IM_SIZE, NUM_CLASSES)


    weights_path = f'models_archive/{args.model_weights_name}'
    print('loading weights from ', weights_path)
    model.load_weights(weights_path)
    weights_loaded = True

    print(f'done! took {time.time()-start_time:.2f} seconds')
    model.summary()  # print model summary

    # log trainable parameters of model
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    run.log('trainable parameters', trainable_count)


    # compile model for evaluation
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    acc_metric = tf.keras.metrics.CategoricalAccuracy()

    model.compile(loss=loss, optimizer=optimizer, metrics=[acc_metric])

    # evaluate against test dataset using model.evaluate
    print("testing against test dataset")
    loss, acc = model.evaluate(test_dataset)
    run.log('test accuracy', np.float(acc))
    run.log('test loss', np.float(loss))
    print('done and logged!')

    # evaluate manually and save predictions
    test_pred, test_true = eval_utils.get_pred_true(model, test_dataset)
    np.savetxt('outputs/test_pred.txt', test_pred)
    np.savetxt('outputs/test_true.txt', test_true)

    test_acc_manual = np.average(test_pred==test_true)
    print('manual test accuracy: ', test_acc_manual)
    run.log('manual test acc', test_acc_manual)

    print('All done!')
