
import os
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.backend import sequence_masking
from bert4keras.backend import recompute_grad
from keras import initializers, activations
from keras.initializers import Constant
from keras.layers import *
from tensorflow_addons.text.crf import crf_decode






def integerize_shape(func):
    """装饰器，保证input_shape一定是int或None
    """
    def convert(item):
        if hasattr(item, '__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item, 'value'):
            return item.value
        else:
            return item

    def new_func(self, input_shape):
        input_shape = convert(input_shape)
        return func(self, input_shape)

    return new_func


if keras.__version__[-2:] != 'tf' and keras.__version__ < '2.3':

    class Layer(keras.layers.Layer):
        """重新定义Layer，赋予“层中层”功能
        （仅keras 2.3以下版本需要）
        """
        def __init__(self, **kwargs):
            super(Layer, self).__init__(**kwargs)
            self.supports_masking = True  # 本项目的自定义层均可mask

        def __setattr__(self, name, value):
            if isinstance(value, keras.layers.Layer):
                if not hasattr(self, '_layers'):
                    self._layers = []
                if value not in self._layers:
                    self._layers.append(value)
            super(Layer, self).__setattr__(name, value)

        @property
        def trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            if trainable:
                trainable_weights = super(Layer, self).trainable_weights[:]
                for l in getattr(self, '_layers', []):
                    trainable_weights += l.trainable_weights
                return trainable_weights
            else:
                return []

        @property
        def non_trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            non_trainable_weights = super(Layer, self).non_trainable_weights[:]
            for l in getattr(self, '_layers', []):
                if trainable:
                    non_trainable_weights += l.non_trainable_weights
                else:
                    non_trainable_weights += l.weights
            return non_trainable_weights

else:

    class Layer(keras.layers.Layer):
        def __init__(self, **kwargs):
            super(Layer, self).__init__(**kwargs)
            self.supports_masking = True  # 本项目的自定义层均可mask

            
class ConditionalRandomField(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层。
    """
    def __init__(self, seq_lr_multiplier=1,tag_lr_multiplier=1,**kwargs):
        super(ConditionalRandomField, self).__init__(**kwargs)
        self.seq_lr_multiplier = seq_lr_multiplier  # 当前层学习率的放大倍数
        self.tag_lr_multiplier = tag_lr_multiplier # 当前层学习率的放大倍数
        self.num=0

    @integerize_shape
    def build(self, input_shape):
        self.log_vars = []
        for i in range(2):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(ConditionalRandomField, self).build(input_shape)
#         print('input_shape:',input_shape)
#         a=input()
        seq_output_dim,tag_output_dim= input_shape[0][-1],input_shape[1][-1]
        self._trans1 = self.add_weight(
            name='trans_seq',
            shape=(seq_output_dim, seq_output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self._trans2 = self.add_weight(
            name='trans_tag',
            shape=(tag_output_dim, tag_output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        if self.seq_lr_multiplier != 1:
            K.set_value(self._trans1, K.eval(self._trans1) / self.seq_lr_multiplier)
        if self.tag_lr_multiplier != 1:
            K.set_value(self._trans2, K.eval(self._trans2) / self.tag_lr_multiplier)

    @property
    def trans_seq(self):
        if self.seq_lr_multiplier != 1:
            return self.seq_lr_multiplier * self._trans1
        else:
            return self._trans1
    @property
    def trans_tag(self):
        if self.tag_lr_multiplier != 1:
            return self.tag_lr_multiplier * self._trans2
        else:
            return self._trans2

    def compute_mask(self, inputs, mask=None):
        return None
    
    
    def multi_loss(self, ys_true, ys_pred):
 
        precision1,precision2 = K.exp(-self.log_vars[0][0]),K.exp(-self.log_vars[1][0])
        loss=(precision1*self.seq_sparse_loss(ys_true[0],ys_pred[0])+self.log_vars[0][0])+(precision2*self.tag_sparse_loss(ys_true[1],ys_pred[1])+self.log_vars[1][0])
#         loss=self.seq_sparse_loss(ys_true[0],ys_pred[0])+self.tag_sparse_loss(ys_true[1],ys_pred[1])
        
#         for y_true, y_pred, log_var ,crf in zip(ys_true, ys_pred, self.log_vars,self.crf_list):
#             precision = K.exp(-log_var[0])
# #             loss += K.sum(precision * crf.sparse_loss(y_true,y_pred) + log_var[0], -1)
#             loss+=(precision * crf.sparse_loss(y_true,y_pred)+ log_var[0])
        return K.mean(loss)
    

    def call(self, inputs):

#         if mask is not None:
#             mask = K.cast(mask, K.floatx())
        
        seq_target,tag_target,seq_true,tag_true=inputs
        loss=self.multi_loss([seq_true,tag_true],[seq_target,tag_target])
        self.add_loss(loss)
        seq_input_shape,tag_input_shape = tf.slice(tf.shape(seq_target), [0], [2]),tf.slice(tf.shape(tag_target), [0], [2])
        seq_mask,tag_mask = tf.ones(seq_input_shape),tf.ones(tag_input_shape)
        seq_sequence_lengths,tag_sequence_lengths = K.sum(K.cast(seq_mask, 'int32'), axis=-1),K.sum(K.cast(tag_mask, 'int32'), axis=-1)
#         seq_length=y_pred.shape[1].value
        seq_target, _ =crf_decode(seq_target,self.trans_seq,seq_sequence_lengths)
        tag_target, _ =crf_decode(tag_target,self.trans_tag,tag_sequence_lengths)
   
        return [seq_target,tag_target]

    """分词的loss和metric"""
    def seq_target_score(self, y_true, y_pred):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        """
        point_score = tf.einsum('bni,bni->b', y_true, y_pred)  # 逐标签得分
        trans_score = tf.einsum(
            'bni,ij,bnj->b', y_true[:, :-1], self.trans_seq, y_true[:, 1:]
        )  # 标签转移得分
        return point_score + trans_score

    def seq_log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans_seq, 0)  # (1, output_dim, output_dim)
        outputs = tf.reduce_logsumexp(
            states + trans, 1
        )  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def seq_dense_loss(self, y_true, y_pred):
        """y_true需要是one hot形式
        """
        # 导出mask并转换数据类型
        mask = K.all(K.greater(y_pred, -1e6), axis=2, keepdims=True)
        mask = K.cast(mask, K.floatx())
        # 计算目标分数
        y_true, y_pred = y_true * mask, y_pred * mask
        target_score = self.seq_target_score(y_true, y_pred)
        # 递归计算log Z
        init_states = [y_pred[:, 0]]
        y_pred = K.concatenate([y_pred, mask], axis=2)
        input_length = K.int_shape(y_pred[:, 1:])[1]
        log_norm, _, _ = K.rnn(
            self.seq_log_norm_step,
            y_pred[:, 1:],
            init_states,
            input_length=input_length
        )  # 最后一步的log Z向量
        log_norm = tf.reduce_logsumexp(log_norm, 1)  # logsumexp得标量
        # 计算损失 -log p
        return log_norm - target_score

    def seq_sparse_loss(self, y_true, y_pred):
        """y_true需要是整数形式（非one hot）
        """
        # y_true需要重新明确一下shape和dtype
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        # 转为one hot
        y_true = K.one_hot(y_true, K.shape(self.trans_seq)[0])

        return self.seq_dense_loss(y_true, y_pred)


    def seq_sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
#         raw_input_shape = tf.slice(tf.shape(y_pred), [0], [2])
#         mask = tf.ones(raw_input_shape)
#         sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)
# #         seq_length=y_pred.shape[1].value
#         y_pred, best_score=crf_decode(y_pred,self.trans,sequence_lengths)
        y_true = tf.cast(y_true,tf.int32)
        y_pred = tf.cast(y_pred,tf.int32)
        v1 = y_true*y_true
        v2 = y_pred*y_true
        values = tf.cast(tf.equal(tf.cast(v1,tf.int32), tf.cast(v2,tf.int32)), tf.int32)
        num = tf.cast(tf.equal(tf.reduce_sum(values,1),tf.reduce_sum(tf.ones_like(values),1)),tf.int32)
        return tf.reduce_sum(num)/tf.shape(y_true)[0]
    
    
    
    
    
    
    
    """标注的loss和metric"""
    def tag_target_score(self, y_true, y_pred):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        """
        point_score = tf.einsum('bni,bni->b', y_true, y_pred)  # 逐标签得分
        trans_score = tf.einsum(
            'bni,ij,bnj->b', y_true[:, :-1], self.trans_tag, y_true[:, 1:]
        )  # 标签转移得分
        return point_score + trans_score

    def tag_log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans_tag, 0)  # (1, output_dim, output_dim)
        outputs = tf.reduce_logsumexp(
            states + trans, 1
        )  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def tag_dense_loss(self, y_true, y_pred):
        """y_true需要是one hot形式
        """
        # 导出mask并转换数据类型
        mask = K.all(K.greater(y_pred, -1e6), axis=2, keepdims=True)
        mask = K.cast(mask, K.floatx())
        # 计算目标分数
        y_true, y_pred = y_true * mask, y_pred * mask
        target_score = self.tag_target_score(y_true, y_pred)
        # 递归计算log Z
        init_states = [y_pred[:, 0]]
        y_pred = K.concatenate([y_pred, mask], axis=2)
        input_length = K.int_shape(y_pred[:, 1:])[1]
        log_norm, _, _ = K.rnn(
            self.tag_log_norm_step,
            y_pred[:, 1:],
            init_states,
            input_length=input_length
        )  # 最后一步的log Z向量
        log_norm = tf.reduce_logsumexp(log_norm, 1)  # logsumexp得标量
        # 计算损失 -log p
        return log_norm - target_score

    def tag_sparse_loss(self, y_true, y_pred):
        """y_true需要是整数形式（非one hot）
        """
        # y_true需要重新明确一下shape和dtype
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        # 转为one hot
        y_true = K.one_hot(y_true, K.shape(self.trans_tag)[0])
        return self.tag_dense_loss(y_true, y_pred)


    def tag_sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
#         raw_input_shape = tf.slice(tf.shape(y_pred), [0], [2])
#         mask = tf.ones(raw_input_shape)
#         sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)
# #         seq_length=y_pred.shape[1].value
#         y_pred, best_score=crf_decode(y_pred,self.trans,sequence_lengths)
        y_true = tf.cast(y_true,tf.int32)
        y_pred = tf.cast(y_pred,tf.int32)
        v1 = y_true*y_true
        v2 = y_pred*y_true
        values = tf.cast(tf.equal(tf.cast(v1,tf.int32), tf.cast(v2,tf.int32)), tf.int32)
        num = tf.cast(tf.equal(tf.reduce_sum(values,1),tf.reduce_sum(tf.ones_like(values),1)),tf.int32)
        return tf.reduce_sum(num)/tf.shape(y_true)[0]
    
    def sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """

        y_true = tf.cast(y_true,tf.int32)
        y_pred = tf.cast(y_pred,tf.int32)
        v1 = y_true*y_true
        v2 = y_pred*y_true
        values = tf.cast(tf.equal(tf.cast(v1,tf.int32), tf.cast(v2,tf.int32)), tf.int32)
        num = tf.cast(tf.equal(tf.reduce_sum(values,1),tf.reduce_sum(tf.ones_like(values),1)),tf.int32)
        return tf.reduce_sum(num)/tf.shape(y_true)[0]

    
    

    

    def get_config(self):
        config = {
            'seq_lr_multiplier': self.seq_lr_multiplier,
            'tag_lr_multiplier': self.tag_lr_multiplier,
        }
        base_config = super(ConditionalRandomField, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
custom_objects = {
    'ConditionalRandomField': ConditionalRandomField,
}

keras.utils.get_custom_objects().update(custom_objects)
