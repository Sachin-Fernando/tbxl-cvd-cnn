import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def focal_loss_fixed(y_true, y_pred):
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
    focal_factor = tf.pow(1 - y_pred, 2.0)
    loss = 0.25 * focal_factor * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
