import tensorflow as tf
import tensorflow.contrib.slim as slim

class DigitsModel:

    def __init__(self, num_classes, learning_rate, img_size=32, batch_decay=0.95):
        self.img_size = img_size
        self.batch_decay = batch_decay
        self.learning_rate = learning_rate
        self.num_classes = num_classes

    def net(self, inputs, is_training):
        with slim.arg_scope([slim.conv2d],
            padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': is_training, 'decay': self.batch_decay}):

            x = slim.conv2d(inputs, 16, [5, 5], scope='conv1')
            x = slim.max_pool2d(x, [2, 2], scope='maxpool1')
            x = slim.conv2d(x, 32, [1, 1], scope='conv2')

            x = slim.conv2d(x, 64, [3, 3], scope='conv3')
            x = slim.max_pool2d(x, [2, 2], padding='VALID', scope='maxpool2')
            x = slim.conv2d(x, 64, [1, 1], scope='conv4')

            x = slim.conv2d(x, 128, [3, 3], scope='conv5')
            x = slim.conv2d(x, self.num_classes, [1, 1], scope='conv6')

        output = slim.avg_pool2d(x, [8, 8], padding='VALID', scope='output')

        return output

    def get_metrics(self, output, labels):

        # make logits to format: [batch_size, num_classes]
        output = tf.squeeze(output)
        logits = tf.nn.softmax(output)

        predictions = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))

        # sparse means only one correct class, it will accelerate the learning process
        losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output))

        return {'logits': logits, 'predictions': predictions, 'accuracy': accuracy,
                'losses': losses}


    def train(self, losses):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        trainer = slim.learning.create_train_op(losses, optimizer)

        return trainer




