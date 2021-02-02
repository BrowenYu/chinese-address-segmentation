from tensorflow.keras.layers import Layer,Conv1D
import tensorflow as tf


class WindowEmbedding(Layer):
    def __init__(self, window_size, **Args):
        super(WindowEmbedding, self).__init__()
        self.window_size = window_size
        self.embedding = tf.keras.layers.Embedding(**Args)
    
    def call(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings_shape = tf.shape(embeddings)
        zero = tf.zeros(shape=(embeddings_shape[0], self.window_size-1, embeddings_shape[2]))
        embeddings = tf.concat([embeddings, zero], axis=1)
        new_embeddings = tf.identity(embeddings)
        for i in range(1, self.window_size):
            zero = tf.zeros(shape=(embeddings_shape[0], i, embeddings_shape[2]))
            temp = embeddings[:, i:, :]
            embeddings_temp = tf.concat([temp, zero], axis=1)
#             new_embeddings = tf.concat([new_embeddings, embeddings_temp], axis=2)
            new_embeddings = tf.add(new_embeddings,embeddings_temp)
        new_embeddings = new_embeddings[:, :embeddings_shape[1], :]
        return new_embeddings
    
    def get_config(self):
        config = {
            'window_size': self.window_size,
        }
        embedding_config = self.embedding.get_config()
        base_config = super(WindowEmbedding, self).get_config()
        return dict(list(config.items()) + list(embedding_config.items()))
    
    @classmethod
    def from_config(cls, config):
        return WindowEmbedding(config['window_size'], 
                               config['input_dim'], 
                               config['output_dim'],
                               config['trainable'],
                               config['mask_zero'])


class WindowEmbeddingforword(Layer):
    def __init__(self, window_size, **Args):
        super(WindowEmbeddingforword, self).__init__()
        self.window_size = window_size
        self.embedding = tf.keras.layers.Embedding(**Args)
    
    def call(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings_shape = tf.shape(embeddings)
        zero = tf.zeros(shape=(embeddings_shape[0], self.window_size-1, embeddings_shape[2]))
        embeddings = tf.concat([zero, embeddings], axis=1)
        new_embeddings = tf.identity(embeddings)
        for i in range(self.window_size-2, -1, -1):
            zero = tf.zeros(shape=(embeddings_shape[0], self.window_size-(i+1), embeddings_shape[2]))
            temp = embeddings[:, :-(self.window_size-(i+1)), :]
            embeddings_temp = tf.concat([zero, temp], axis=1)
            new_embeddings = tf.concat([new_embeddings, embeddings_temp], axis=2)
        new_embeddings = new_embeddings[:, self.window_size-1:, :]
        return new_embeddings
    
    def get_config(self):
        config = {
            'window_size': self.window_size,
        }
        embedding_config = self.embedding.get_config()
        base_config = super(WindowEmbeddingforword, self).get_config()
        return dict(list(config.items()) + list(embedding_config.items()))
    
    @classmethod
    def from_config(cls, config):
        return WindowEmbeddingforword(config['window_size'], 
                                      config['input_dim'], 
                                      config['output_dim'],
                                      config['trainable'],
                                      config['mask_zero'])
    
    
class IDCNN(Layer):
    def __init__(self, repeat_times=1, dilations=[1,1,2],filters=[256,256,512],kernel_size=[2,3,4],**kwargs):
        self.repeat_times=repeat_times
        self.dilation = dilations
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1d_list = [[Conv1D(filters=self.filters[j],
                   kernel_size=self.kernel_size[j],
                   activation='relu',
                   padding='same',
                   dilation_rate=self.dilation[j])for j in range(3)] for i in range(self.repeat_times)]
        
        super(IDCNN, self).__init__(**kwargs)
        

    def call(self, inputs):
        for i in range(self.repeat_times):
            for j in range(3):
                inputs=self.conv1d_list[i][j](inputs)
        return inputs    