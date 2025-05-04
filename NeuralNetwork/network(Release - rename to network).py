from tensorflow.keras.metrics import Metric # type: ignore

import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
def weightedBinaryCrossentropy(classWeights):
    """
    returns a custom loss function with class specific weights.
    """
    classWeights = tf.constant(classWeights, dtype=tf.float32)

    @keras.saving.register_keras_serializable()
    def loss_fn(y_true, y_pred):
        # clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # apply the binary cross entropy formula with the weights
        bce = -(classWeights * y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred))

        return tf.reduce_mean(bce)  # average over batch and classes

    return loss_fn

@keras.saving.register_keras_serializable()
class F1Score(Metric):
    '''
    custom class inhereting the keras Metric class
    calculated the F score of a model
    '''
    def __init__(self, num_classes=3, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # predictions to binary w threshold 0.5
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # flatten to 2d
        if len(y_pred.shape) == 3:
            y_pred = tf.reshape(y_pred, [-1, self.num_classes])
            y_true = tf.reshape(y_true, [-1, self.num_classes])

        # true and false positives, and false negatives
        tp = tf.reduce_sum(y_pred * y_true)
        fp = tf.reduce_sum(y_pred * (1 - y_true))
        fn = tf.reduce_sum((1 - y_pred) * y_true)
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return f1

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

    # the f score formula
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        return 2 * precision * recall / (precision + recall + 1e-7)

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)