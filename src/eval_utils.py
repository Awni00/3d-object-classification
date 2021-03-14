import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def get_gradients(inputs, y, model):
    '''get gradients of model's output w.r.t. input images'''

    with tf.GradientTape() as tape:
        # format dims
        inputs = format_dims(inputs)

        # cast as float
        inputs = [tf.cast(input_, tf.float32) for input_ in inputs]

        # watch the input pixels
        tape.watch(inputs)

        # generate predictions
        predictions = model(inputs)

        # get the loss
        loss = tf.keras.losses.categorical_crossentropy(y, predictions)

    # get the gradient with respect to the inputs
    gradients = tape.gradient(loss, inputs)

    return gradients

def gen_img_saliency(img, grads, plot=False):
    '''generate saliency map for one image'''

    # reduce the RGB image to grayscale
    grayscale_tensor = tf.reduce_sum(tf.abs(grads), axis=-1)

    # normalize the pixel values to be in the range [0, 255].
    # the max value in the grayscale tensor will be pushed to 255.
    # the min value will be pushed to 0.
    normalized_tensor = tf.cast(
        255 * (grayscale_tensor - tf.reduce_min(grayscale_tensor))
        / (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)),
        tf.uint8)

    # remove the channel dimension to make the tensor a 2d tensor
    normalized_tensor = tf.squeeze(normalized_tensor)

    if plot:
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(normalized_tensor, cmap='gray')
        plt.show()

    return normalized_tensor


def gen_super_imposed(img, saliency, plot=False):
    '''superimposed original image with its saliency map'''

    gradient_color = cv2.applyColorMap(np.array(saliency), cv2.COLORMAP_HOT)
    gradient_color = np.array(gradient_color / 255.0)
    super_imposed = cv2.addWeighted(np.array(img) / 255., 0.5, gradient_color, 0.5, 0.0, dtype = cv2.CV_32F)

    if plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(super_imposed)
        plt.axis('off')
        plt.show()

    return super_imposed


def gen_saliency_map(inputs, y, model, plot=True):
    '''generate saliency maps for (rgb, d) image pair'''

    gradients = get_gradients(inputs, y, model)

    rgb_img, d_img = inputs
    rgb_grads, d_grads = gradients

    rgb_sal = gen_img_saliency(rgb_img, rgb_grads, plot=False)
    d_sal = gen_img_saliency(d_img, d_grads, plot=False)

    rgb_superimposed = gen_super_imposed(rgb_img[0], rgb_sal)
    d_superimposed = gen_super_imposed(d_img[0], d_sal)

    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6));
        ax1.imshow(rgb_superimposed);
        ax2.imshow(d_superimposed);
        return fig
    else:
        return (rgb_superimposed, d_superimposed)

