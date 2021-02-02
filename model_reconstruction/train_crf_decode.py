# TensorFlow2.3
# python3.8
# hanlp v2.0.0-alpha.0

import os, datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# 设置为-1表示不实用GPU，GPU加速效果明显
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_KERAS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from datetime import datetime

import math

import bert4keras
from bert4keras.backend import keras, K
from hanlp.common.structure import SerializableDict
from hanlp.transform.tsv import TSVTaggingTransform
from hanlp.common.vocab import Vocab
from hanlp.optimizers.adamw.optimization import AdamWeightDecay
from hanlp.utils import io_util
from hanlp.common.transform import Transform

from bert4keras.snippets import ViterbiDecoder, to_array
from crf_layer import ConditionalRandomField
# from bert4keras.layers import ConditionalRandomField
# from tensorflow.contrib.crf import crf_decode

import tensorflow as tf 
import numpy as np
import pandas as pd 
# import matplotlib.pyplot as plt

from tensorflow.keras import models,layers,losses,callbacks,optimizers


PATH = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".") + '/data/'

trn_path = PATH+'generate/train_87w.txt'
dev_path = PATH+'generate/dev_87w.txt'
save_model='model/'       #存储model
save_dir = 'save_dir/'    #存储vacab.json等初始化文件
all_data_len = 852480

class SparseAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='Sparse_accuracy',**kwargs):
        super(SparseAccuracy,self).__init__(name=name,**kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
#         raw_input_shape = tf.slice(tf.shape(y_pred), [0], [2])
#         mask = tf.ones(raw_input_shape)
#         sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)
#         seq_length=y_pred.shape[1].value
        y_pred, best_score=crf_decode(y_pred,self.trans,sequence_lengths)
        y_true = tf.cast(y_true,tf.int32)
        y_pred = tf.cast(y_pred,tf.int32)
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

# @tf.function
def build_model(transform, embeddings=100, embedding_trainable=False,rnn_input_dropout=0.2,
            rnn_output_dropout=0.2, rnn_units=100,crf_lr_multiplier=100,**kwargs) -> tf.keras.Model:
    model = tf.keras.Sequential()
    
    embeddings = tf.keras.layers.Embedding(input_dim=len(transform.word_vocab), output_dim=embeddings,
                                            trainable=True, mask_zero=True)
    model.add(embeddings)
    if rnn_input_dropout:
        model.add(tf.keras.layers.Dropout(rnn_input_dropout, name='rnn_input_dropout'))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=rnn_units, return_sequences=True), name='bilstm'))
    if rnn_output_dropout:
        model.add(tf.keras.layers.Dropout(rnn_output_dropout, name='rnn_output_dropout'))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(transform.tag_vocab)), name='dense'))
    CRF=ConditionalRandomField(lr_multiplier=crf_lr_multiplier,name='crf')
    model.add(CRF)
    return {'model':model,'CRF':CRF}


def build(transform):
    model,CRF = build_model(transform,embeddings=100, embedding_trainable=True,
            rnn_input_dropout=0.5, rnn_output_dropout=0.5, rnn_units=300,crf_lr_multiplier=1e-2)['model'],build_model(transform,embeddings=100, embedding_trainable=True,
            rnn_input_dropout=0.5, rnn_output_dropout=0.5, rnn_units=300,crf_lr_multiplier=1e-2)['CRF']

    x_shape = [None]
    model.build(input_shape=x_shape)

    def build_optimizer(optimizer):
        if isinstance(optimizer, (str, dict)):
            custom_objects = {'AdamWeightDecay': AdamWeightDecay}
            optimizer: tf.keras.optimizers.Optimizer = tf.keras.utils.deserialize_keras_object(optimizer,
                                                                                               module_objects=tf.keras.optimizers,
                                                                                               custom_objects=custom_objects)
        return optimizer
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) 
    optimizer = build_optimizer('Adam') 
    #build_optimizer('Adam')
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=losses.Reduction.SUM,from_logits=True)
#     metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),SparseAccuracy(Crf_trans=CRF.trans)]
    metrics=[CRF.sparse_accuracy]
    model.compile(optimizer=optimizer, loss=CRF.sparse_loss, metrics=metrics, run_eagerly=None)

    tf.print(model.summary())
    return model

def main():
    batch_size = 512
    epochs = 2000
    train_steps_per_epoch = ( all_data_len*0.8 ) // batch_size
    dev_steps = (all_data_len*0.2) // batch_size


    
    transform = TSVTaggingTransform()
    # 读取字典
    load_vocabs(transform, save_dir)
    # 构建模型
    model=build(transform)

    # 把训练数据和验证数据转为tf.data.Dataset格式
    trn_data=transform.file_to_dataset(trn_path, batch_size=batch_size,shuffle=True,repeat=-1)
    dev_data = transform.file_to_dataset(dev_path, batch_size=batch_size, shuffle=True,repeat=-1)

#     tf.print('Count dataset size...')
#     train_steps_per_epoch = math.ceil(size_of_dataset(trn_data) / batch_size)
#     dev_steps = math.ceil(size_of_dataset(dev_data) / batch_size)
    
#     tf.print(f'train_steps_per_epoch: {train_steps_per_epoch}')
#     tf.print(f'dev_steps: {dev_steps}')
    
    # 设立指标，存储指标最优点的模型
    metrics="sparse_accuracy"
    monitor = f'val_{metrics}'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(save_model, '0122_model_demo1.h5'),
                # verbose=1,
                monitor=monitor, save_best_only=True,
                mode='max',
                save_weights_only=True)
    
#     early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=15)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=io_util.makedirs(io_util.path_join(save_model, 'logs_0122_demo1')))
    
    callbacks = [checkpoint, tensorboard_callback]
    
    # 模型训练
#     tf.debugging.set_log_device_placement(False)
    history = model.fit(trn_data, epochs=epochs, steps_per_epoch=train_steps_per_epoch,
                         validation_data=dev_data,
                         callbacks=callbacks,
                         validation_steps=dev_steps,verbose=1)



if __name__ == "__main__":
    main()