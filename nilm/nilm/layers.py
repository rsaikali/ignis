"""
Custom Keras layers for NILM models.
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class MultiHeadAttentionLayer(layers.Layer):
    """
    Multi-Head Attention pour mieux capturer les patterns de multiples
    appliances simultanés.
    """

    def __init__(self, num_heads=4, key_dim=32, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        """
        Args:
            inputs: (batch, sequence, features)

        Returns:
            Attended features de même shape
        """
        return self.attention(inputs, inputs)

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config
