import tensorflow as tf

# temporarily setting default value for c=1/5.
# We will have to calculate this on a per batch basis
# but it might make sense to do like this for now.
def huber_loss(y_pred, y_actual, c):
    x = tf.subtract(y_pred, y_actual)
    abs = tf.abs(x)
    quadratic = tf.square(abs) + tf.square(c)
    quadratic = tf.divide(quadratic, 2 * c)
    return tf.where(abs < c, abs, quadratic)
