import tensorflow as tf
import numpy as np

import utils

def predict_class(model, input_, class_dict=None, y=None):
    '''predicts the class of the given input'''

    input_ = format_dims(input_)

    pred = model.predict(input_)
    pred_class = np.argmax(pred, axis=-1)

    if class_dict:
        return [class_dict[p] for p in pred_class]
    else:
        return pred_class

def format_dims(input_):
    """
    adds batch dimension to tensor or array of tensors if necessary (i.e.: if tensor.shape==3 else)

    Args:
        input_ (tf.Tensor or List): image or list of images as tensors

    Raises:
        ValueError: input not tf.Tensor or list of tf.Tensor's

    Returns:
        tf.Tensor or List: input with added batch dimension (if applicable)
    """

    if isinstance(input_, list):
        output = []
        for tens in input_:
            if len(tens.shape) == 3:
                output.append(tf.expand_dims(tens, axis=0))
            else:
                output.append(tens)

        return output

    elif isinstance(input_, tf.Tensor) or isinstance(input_, np.ndarray):
        if len(input_.shape) == 3:
            return tf.expand_dims(input_, axis=0)
        else:
            return input_

    else:
        raise ValueError('failed to expand dims. Input not tensor or list of tensors')

def plot_and_predict(model, rgb, d, y=None, int_label_dict=None):
    '''gets model's prediction on rgb-d image and plots the image.'''

    utils.plot_rgb_depth(rgb, d)

    [rgb, d] = format_dims([rgb,d])

    pred = model.predict([rgb, d])
    pred_class = np.argmax(pred, axis=-1)[0]

    if int_label_dict:
        print(f'{pred_class}; {int_label_dict[pred_class]}')

def get_pred_true(model, batched_dataset):
    """
    generates the predicted classes and ground truth classes of a dataset and model.

    Args:
        model: model w/ a predict method and returns softmax predictions
        batched_dataset: a batched zip dataset of the form (input, output) where output is OHE.

    Returns:
        2-tuple of np.ndarray: tuple of predicted_classes, true_classes
    """

    eval_preds = np.array([])
    eval_true = np.array([])

    print('generating (preds, true) for dataset: ')

    num_batches = len(batched_dataset)
    for i, (x, y) in zip(range(1, num_batches+1), batched_dataset):
        print(f'    [{i}/{num_batches}]')

        preds = model.predict(x)
        pred_classes = np.argmax(preds, axis=-1)

        true_classes = np.argmax(y, axis=-1)

        eval_preds = np.concatenate([eval_preds, pred_classes])
        eval_true = np.concatenate([eval_true, true_classes])

    return eval_preds, eval_true
