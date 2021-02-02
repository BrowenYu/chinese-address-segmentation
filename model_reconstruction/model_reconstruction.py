# TensorFlow2.3
# python3.8
# hanlp v2.0.0-alpha.0

import os, datetime
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# # 设置为-1表示不实用GPU，GPU加速效果明显
os.environ["CUDA_VISIBLE_DEVICES"]="0"     
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import models,layers,losses,callbacks,optimizers
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dropout, TimeDistributed, Dense
from tensorflow.keras.models import Model
from hanlp.common.structure import SerializableDict
from hanlp.transform.tsv import TSVTaggingTransform
from hanlp.common.vocab import Vocab
from hanlp.optimizers.adamw.optimization import AdamWeightDecay
from hanlp.utils import io_util
from hanlp.common.transform import Transform

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from layers import WindowEmbedding, WindowEmbeddingforword


trn_path = 'dataset/data0129_bidata/train0129_all.txt'
dev_path = 'dataset/data0129_bidata/dev0129_all.txt'
save_model='model_dir/0129_best_model/'       #存储model
save_dir = 'save_dir/'    #存储vacab.json等初始化文件
static_model_path = './static_model/model/'

class SparseAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='Sparse_accuracy', **kwargs):
        super(SparseAccuracy,self).__init__(name=name,**kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true,tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred,axis=-1),tf.int32)
        v1 = y_true*y_true
        v2 = y_pred*y_true
        values = tf.cast(tf.equal(tf.cast(v1,tf.int32), tf.cast(v2,tf.int32)), tf.int32)
        num = tf.cast(tf.equal(tf.reduce_sum(values,1),tf.reduce_sum(tf.ones_like(values),1)),tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(num))

    @tf.function
    def result(self):
        return self.count/self.total


def size_of_dataset(dataset: tf.data.Dataset) -> int:
    count = 0
    for element in dataset.unbatch().batch(1):
        count += 1
        tf.print(count)
    return count


def load_vocabs(transform,save_dir,filename='vocabs.json'):
    vocabs = SerializableDict()
    vocabs.load_json(os.path.join(save_dir, filename))
    for key, value in vocabs.items():
        vocab = Vocab()
        vocab.copy_from(value)
        setattr(transform, key, vocab)


def scheduler(epoch, lr):
  if epoch < 50:
    return lr
  else:
    return lr * math.exp(-0.01)


def build_model(transform, embeddings_size=100, rnn_units=100, window_size=1, **kwargs) -> tf.keras.Model:
    S_inputs = Input(shape=(None,), dtype='int32')
    embedding = tf.keras.layers.Embedding(input_dim=len(transform.word_vocab), 
                                          output_dim=embeddings_size,
                                          trainable=True, 
                                          mask_zero=True,)(S_inputs)
    # 静态embedding层
    # static_model = tf.keras.models.load_model(static_model_path, compile=False)
    # static_embeddings_param = static_model.get_weights()[0]
    # static_embedding = StaticEmbed(static_embeddings_param, 
    #                                input_dim=len(transform.word_vocab), 
    #                                embedding_size=embeddings_size, 
    #                                mask_zero=True)(S_inputs)
    # static_embedding = tf.concat([embedding, static_embedding], axis=2)

    # window_embedding层
    window_embeddings = WindowEmbeddingforword(window_size)(embedding)

    dropout_in = Dropout(0.5)(window_embeddings)
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=rnn_units, 
                                                                return_sequences=True,), name="bilstm")(dropout_in)
    dropout_out = Dropout(0.5)(bilstm)
    outputs = TimeDistributed(Dense(len(transform.tag_vocab)), name='dense')(dropout_out)
    model = Model(inputs=S_inputs, outputs=outputs)
    return model


def build(transform):
    model = build_model(transform,embeddings=100, embedding_trainable=True,
            rnn_input_dropout=0.5, rnn_output_dropout=0.5, rnn_units=300, window_size=4)

    x_shape = [None]
    model.build(input_shape=x_shape)

    def build_optimizer(optimizer):
        if isinstance(optimizer, (str, dict)):
            custom_objects = {'AdamWeightDecay': AdamWeightDecay(learning_rate=0.001,
                                                                weight_decay_rate=0.01,
                                                                beta_1=0.9,
                                                                beta_2=0.999,
                                                                epsilon=1e-6,
                                                                exclude_from_weight_decay=['layer_norm', 'bias'])}
            
            optimizer: tf.keras.optimizers.Optimizer = tf.keras.utils.deserialize_keras_object(optimizer,
                                                                                            module_objects=tf.keras.optimizers,
                                                                                            custom_objects=custom_objects)
            # optimizer: tf.keras.optimizers.Optimizer = tf.keras.utils.deserialize_keras_object(optimizer)
        return optimizer

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) #build_optimizer('Adam')
    optimizer = build_optimizer("AdamWeightDecay")
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=losses.Reduction.SUM,from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),SparseAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=None)

    tf.print(model.summary())
    return model


def main():
    batch_size = 512
    epochs = 2000

    transform = TSVTaggingTransform()
    # 读取字典
    load_vocabs(transform, save_dir)
    # 构建模型
    # strategy = tf.distribute.MirroredStrategy(devices=['/device:GPU:0', '/device:GPU:1'])
    # with strategy.scope():
    #     model = build(transform)
    model = build(transform)

    # 把训练数据和验证数据转为tf.data.Dataset格式
    trn_data=transform.file_to_dataset(trn_path, batch_size=batch_size,shuffle=False,repeat=1)
    dev_data = transform.file_to_dataset(dev_path, batch_size=batch_size, shuffle=True,repeat=1)
    
    # 设立指标，存储指标最有点的模型
    metrics="Sparse_accuracy"
    monitor = f'val_{metrics}'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                save_model,
                # verbose=1,
                monitor=monitor, save_best_only=True,
                mode='max',
                save_weights_only=False)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                            log_dir=io_util.makedirs(io_util.path_join(save_model, 'logs')))
    earlystop = tf.keras.callbacks.EarlyStopping(monitor, mode='max', patience=50)
    csvlogger = tf.keras.callbacks.CSVLogger("0129train.csv")
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    callbacks = [checkpoint, tensorboard_callback, earlystop, csvlogger, learning_rate_scheduler]
    
    model.save('./model_struct/0129_best_model/', save_format='tf')
    # 模型训练
    history = model.fit(trn_data, epochs=epochs,
                         validation_data=dev_data,
                         callbacks=callbacks,verbose=1)

if __name__ == "__main__":
    main()
