import tensorflow as tf

def train_epoch(model, train_dataset, epoch_num, optimizer, loss, train_metric, max_steps=None, logger=None):
    '''train model for one epoch'''

    print('\n')
    print('==============')
    print(f'EPOCH {epoch_num}')
    print('==============')

    if not max_steps: max_steps = len(train_dataset)

    train_iter = train_dataset.__iter__()
    losses = []

    step = 1
    while step <= max_steps:
        try:
            (x_batch, y_batch) = train_iter.next()

            logits, loss_value = apply_gradient(model, optimizer, loss, x_batch, y_batch)
            losses.append(loss_value)

            # accumulate metrics
            train_metric.update_state(y_batch, logits)

        except Exception as e:
            print(f'INFO: failed to get data for epoch {epoch_num} step {step}. skipping...')
            if logger: logger.error(f'failed to get data for train epoch {epoch_num} step {step}')

            print(e)


        print(f'step {step:03} / {max_steps}')

        step += 1

    return losses


def apply_gradient(model, optimizer, loss, x, y):
    '''apply gradients to batch of data'''

    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss(y, logits)

    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return logits, loss_value

def validate_epoch(model, test_dataset, epoch_num, loss, test_metric, max_steps=None, logger=None):
    '''validate mode against test_dataset after epoch of training'''

    losses = []

    print('\n')
    print('validating on test dataset')

    if not max_steps: max_steps = len(test_dataset)

    test_iter = test_dataset.__iter__()
    losses = []

    step = 1
    while step <= max_steps:
        try:
            x_batch, y_batch = test_iter.next()

            val_logits = model(x_batch)
            loss_value = loss(y_batch, val_logits)
            losses.append(loss_value)

            # accumulate metrics
            test_metric.update_state(y_batch, val_logits)

        except:
            print(f'INFO: failed to get data for epoch {epoch_num} step {step}')
            if logger: logger.error(f'failed to get data for test epoch {epoch_num} step {step}')


        print(f'step {step:03} / {max_steps}')

        step += 1


    return losses