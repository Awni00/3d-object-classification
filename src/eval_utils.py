import tensorflow as tf
import numpy as np

import utils

def predict_class(model, input_, class_dict=None, y=None):
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
    utils.plot_rgb_depth(rgb, d)

    [rgb, d] = format_dims([rgb,d])

    pred = model.predict([rgb, d])
    pred_class = np.argmax(pred, axis=-1)[0]

    if int_label_dict:
        print(f'{pred_class}; {int_label_dict[pred_class]}')