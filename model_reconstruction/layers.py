from tensorflow.keras.layers import Layer
import tensorflow as tf


class WindowEmbeddingforword(Layer):
    def __init__(self, window_size, **kwargs):
        super(WindowEmbeddingforword, self).__init__(**kwargs)
        self.window_size = window_size

    @classmethod
    def from_config(cls, config):
        return WindowEmbeddingforword(config['window_size'], config['input_dim'], config['output_dim'], config['trainable'], config['mask_zero'])

    def call(self, inputs):
        embeddings = inputs
        embeddings_shape = tf.shape(embeddings)
        zero = tf.zeros(shape=(embeddings_shape[0], self.window_size-1, embeddings_shape[2]))
        embeddings = tf.concat([zero, embeddings], axis=1)
        new_embeddings = tf.identity(embeddings)
        for i in range(1, self.window_size):
            zero = tf.zeros(shape=(embeddings_shape[0], i, embeddings_shape[2]))
            temp = embeddings[:, :-i, :]
            embeddings_temp = tf.concat([zero, temp], axis=1)
            new_embeddings = tf.concat([new_embeddings, embeddings_temp], axis=2)

        new_embeddings = new_embeddings[:, self.window_size-1:, :]
        return new_embeddings

    def get_config(self):
        config = {
            'window_size': self.window_size,
        }
        base_config = super(WindowEmbeddingforword, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return WindowEmbeddingforword(config['window_size'])


class WindowEmbedding(Layer):
    def __init__(self, window_size, **kwargs):
        super(WindowEmbedding, self).__init__(**kwargs)
        self.window_size = window_size
    
    def call(self, inputs):
        embeddings = inputs
        embeddings_shape = tf.shape(embeddings)
        zero = tf.zeros(shape=(embeddings_shape[0], self.window_size-1, embeddings_shape[2]))
        embeddings = tf.concat([embeddings, zero], axis=1)
        new_embeddings = tf.identity(embeddings)
        for i in range(1, self.window_size):
            zero = tf.zeros(shape=(embeddings_shape[0], i, embeddings_shape[2]))
            temp = embeddings[:, i:, :]
            embeddings_temp = tf.concat([temp, zero], axis=1)
            new_embeddings = tf.concat([new_embeddings, embeddings_temp], axis=2)

        new_embeddings = new_embeddings[:, :embeddings_shape[1], :]
        return new_embeddings
    
    def get_config(self):
        config = {
            'window_size': self.window_size,
        }
        base_config = super(WindowEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return WindowEmbedding(config['window_size'])


class WindowEmbeddingBothSides(Layer):
    def __init__(self, window_size, **kwargs):
        super(WindowEmbedding, self).__init__(**kwargs)
        self.window_size = window_size

    @classmethod
    def from_config(cls, config):
        return WindowEmbeddingBothSides(config['window_size'], config['input_dim'], config['output_dim'], config['trainable'], config['mask_zero'])

    def call(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings_shape = tf.shape(embeddings)
        zero = tf.zeros(shape=(embeddings_shape[0], self.window_size-1, embeddings_shape[2]))
        embeddings = tf.concat([zero, embeddings], axis=1)
        new_embeddings = tf.identity(embeddings)
        for i in range(1, self.window_size):
            zero = tf.zeros(shape=(embeddings_shape[0], i, embeddings_shape[2]))
            temp_left = embeddings[:, :-i, :]
            embeddings_temp_left = tf.concat([zero, temp_left], axis=1)

            temp_right = embeddings[:, i:, :]
            embeddings_temp_right = tf.concat([temp_right, zero], axis=1)

            new_embeddings = tf.concat([embeddings_temp_left, new_embeddings, embeddings_temp_right], axis=2)

        new_embeddings = new_embeddings[:, self.window_size-1:, :]
        return new_embeddings

    def get_config(self):
        config = {
            'window_size': self.window_size,
        }
        base_config = super(WindowEmbeddingBothSides, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return WindowEmbeddingBothSides(config['window_size'])


class StaticEmbed(Layer):
    def __init__(self, static_embeddings_param, input_dim, embedding_size, mask_zero=True, **kwargs):
        self.static_embeddings_param = static_embeddings_param
        self.static_embed_param_constant = tf.keras.initializers.Constant(self.static_embeddings_param)
        self.input_dim = input_dim
        self.embedding_size = embedding_size
        self.mask_zero = mask_zero
        self.static_embeddings = tf.keras.layers.Embedding(input_dim=self.input_dim,
                                                           output_dim=self.embedding_size, 
                                                           mask_zero=self.mask_zero,
                                                           trainable=False, 
                                                           embeddings_initializer=self.static_embed_param_constant)
        super(StaticEmbed, self).__init__(**kwargs)

    def call(self, inputs):
        static_embed = self.static_embeddings(inputs)
        return static_embed

    def get_config(self):
        config = {
            'static_embeddings_param': self.static_embeddings_param,
            'input_dim': self.input_dim,
            'embedding_size': self.embedding_size,
            'mask_zero': self.mask_zero,
        }
        base_config = super(StaticEmbed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return StaticEmbed(config['static_embeddings_param'], 
                           config['input_dim'],
                           config['embedding_size'],
                           config['mask_zero'])