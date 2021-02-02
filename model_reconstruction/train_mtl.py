import os
import pickle
from tqdm import tqdm
PATH = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".") + '/data/'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_KERAS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
# import keras
import sys
import bert4keras
import numpy as np
import random as rd
from bert4keras.backend import keras, K
# from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
# from mylayer import ConditionalRandomField
from keras.layers import *
from keras.models import Model
from keras.layers import Lambda
from keras.callbacks import *
# from keras import backend as K
from tqdm import tqdm


epochs = 500
batch_size = 1024
# bert_layers = 12
learing_rate = 1e-4  # bert_layers越小，学习率应该要越大
seq_crf_lr_multiplier = 1e-2  # 必要时扩大CRF层的学习率
tag_crf_lr_multiplier = 1e-2
vocab_size=21128

# bert配置
# config_path = '../../Q_A/publish/bert_config.json'
# checkpoint_path = '../../Q_A/publish/bert_model.ckpt'
dict_path = '../../Q_A/publish/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=False)



labels = ['O','R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R20', 'R21', 'R22', 'R23', 'R24', 
              'R25', 'R30', 'R31', 'R90', 'R99','X']
seg_labels=['O','B','I','E']

id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels)

id2seglabel = dict(enumerate(seg_labels))
seglabel2id = {j: i for i, j in id2seglabel.items()}
num_seglabels = len(seg_labels)

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    # ML = maxlen
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])



class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_seq_labels, batch_tag_labels = [], [], []
        for is_end, (address,tags) in self.sample(random):
            token_ids, seq_labels,tag_labels = [], [], []
            for add,tag in zip(address,tags):
                w_token_ids = tokenizer.encode(add)[0][1:-1]
                token_ids += w_token_ids
                tag_labels+=([label2id[tag]]+[label2id['X']]*(len(w_token_ids)-1))
                B=seglabel2id['B']
                I=seglabel2id['I']
                E=seglabel2id['E']
                if len(w_token_ids)>1:
                    seq_labels+=([B]+[I]*(len(w_token_ids)-2)+[E])
                else:
                    seq_labels+=([B])
            

            
            
            batch_token_ids.append(token_ids)
            batch_seq_labels.append(seq_labels)
            batch_tag_labels.append(tag_labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = seq_padding(batch_token_ids)
                batch_seq_labels = seq_padding(batch_seq_labels)
                batch_tag_labels = seq_padding(batch_tag_labels)
                yield batch_token_ids, [batch_seq_labels,batch_tag_labels]
                batch_token_ids, batch_seq_labels, batch_tag_labels = [], [], []



                
                
def get_valid(valid_data):
    batch_token_ids, batch_seq_labels, batch_tag_labels = [], [], []
    for address,tags in tqdm(valid_data):
        token_ids, seq_labels,tag_labels = [], [], []
        for add,tag in zip(address,tags):
            w_token_ids = tokenizer.encode(add)[0][1:-1]
            token_ids += w_token_ids
            tag_labels+=([label2id[tag]]+[label2id['X']]*(len(w_token_ids)-1))
            B=seglabel2id['B']
            I=seglabel2id['I']
            E=seglabel2id['E']
            if len(w_token_ids)>1:
                seq_labels+=([B]+[I]*(len(w_token_ids)-2)+[E])
            else:
                seq_labels+=([B])
            

        batch_token_ids.append(token_ids)
        batch_seq_labels.append(seq_labels)
        batch_tag_labels.append(tag_labels)
        
    return [batch_token_ids,batch_seq_labels,batch_tag_labels]
    


class SparseAccuracy(keras.metrics.Metric):
    def __init__(self, name='Sparse_accuracy', **kwargs):
        super(SparseAccuracy,self).__init__(name=name,**kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
#         print('shape',K.shape(y_true[0]))
#         print('shape',K.shape(y_true[1]))
#         print('shape',K.shape(y_pred[0]))
#         print('shape',K.shape(y_pred[1]))
#         seg_true,tag_true=y_true[0],y_true[1]
#         seg_pred,tag_pred=y_pred[0],y_true[1]

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


def build_model(embeddings=100,vocab_size=vocab_size,rnn_units=100):
   
    x_in = Input(shape=(None,))
    output=Embedding(input_dim=vocab_size, output_dim=embeddings,
                                            trainable=True, mask_zero=True)(x_in)
    
    flstm_output=LSTM(units=rnn_units, return_sequences=True)(output)
    seq_output=Dropout(0.5)(flstm_output)
    seq_output=TimeDistributed(Dense(num_seglabels), name='dense_seq')(seq_output)
    
    Seq_crf = ConditionalRandomField(lr_multiplier=seq_crf_lr_multiplier,name='seq_crf')
    
    seq_output=Seq_crf(seq_output)
    
    
    reverse_output=Lambda(lambda x: K.reverse(x,axes=1))(output)
    
    reverse_output=LSTM(units=rnn_units, return_sequences=True)(reverse_output)
    
    blstm_output=Lambda(lambda x: K.reverse(x,axes=1))(reverse_output)
    
    lstm_out=Concatenate()([flstm_output,blstm_output])
    tag_output=Dropout(0.5)(lstm_out)
    tag_output=TimeDistributed(Dense(num_labels), name='dense_tag')(tag_output)
    
    Tag_crf = ConditionalRandomField(lr_multiplier=tag_crf_lr_multiplier,name='tag_crf')
    
    tag_output=Tag_crf(tag_output)
    
    

    model = Model(x_in, [seq_output,tag_output])
    model.summary()

    model.compile(
        loss=[Seq_crf.sparse_loss,Tag_crf.sparse_loss],
        optimizer=Adam(learing_rate),
        metrics=[SparseAccuracy()]
    )
    return model,Seq_crf,Tag_crf


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text):
        pass
    
    


def evaluate(data,Seq_ner,Tag_ner,model):
    """评测函数
    """
    token_list,seq_list,tag_list=data
    X, Y, Z= 1e-10, 1e-10,1e-10
    for token, seq , tag in tqdm(zip(token_list,seq_list,tag_list)):
        token_ids= to_array([token])
        P = model.predict([token_ids])
        S,T=list(Seq_ner.decode(P[0][0])),list(Tag_ner.decode(P[1][0]))
        X +=1 if (S==seq) else 0
        Y +=1 if (T==tag) else 0
        Z +=1
    seq_acc, tag_acc = X / Z, Y / Z
    return seq_acc, tag_acc


class Evaluator(keras.callbacks.Callback):
    def __init__(self, valid_data, model, Seq_ner, Tag_ner):
        self.best_val_seq = 0
        self.best_val_tag = 0
        self.valid_data = valid_data
        self.model=model
        self.Seq_ner=Seq_ner
        self.Tag_ner=Tag_ner

    def on_epoch_end(self, epoch, logs=None):
        seq_acc, tag_acc = evaluate(self.valid_data,self.Seq_ner,self.Tag_ner,self.model)
        # 保存最优
        if seq_acc >= self.best_val_seq:
            self.best_val_seq = seq_acc
            self.model.save_weights('./model/best_seq_0111.h5')
        if tag_acc >= self.best_val_tag:
            self.best_val_tag = tag_acc
            self.model.save_weights('./model/best_tag_0111.h5')
        print(
            'valid:  seq_acc: %.5f, tag_acc: %.5f, best_seq_acc: %.5f, best_tag_acc: %.5f\n' %
            (seq_acc, tag_acc, self.best_val_seq, self.best_val_tag)
        )

               
train_path=PATH+'generate/pkl/train_all.pkl'
train_file=pickle.load(open(train_path,'rb'))
# train_len=int(0.1*len(train_file))
# train_file=train_file[:train_len]
np.random.shuffle(train_file)
val_len=int(0.8*len(train_file))
valid_data=get_valid(train_file[-50000:])
# print(valid_data[:10])
train_generator = data_generator(train_file[:val_len], batch_size=batch_size)
valid_generator=data_generator(train_file[val_len:],batch_size=batch_size)



model,Seq_crf,Tag_crf = build_model(embeddings=200,vocab_size=vocab_size,rnn_units=300)
Seq_ner = NamedEntityRecognizer(trans=K.eval(Seq_crf.trans))
Tag_ner = NamedEntityRecognizer(trans=K.eval(Tag_crf.trans))


evaluator = Evaluator(valid_data,model,Seq_ner,Tag_ner)
early_stopping = EarlyStopping(monitor='val_tag_crf_Sparse_accuracy', patience=10)  # 早停法，防止过拟合
plateau = ReduceLROnPlateau(monitor='val_tag_crf_Sparse_accuracy', verbose=1, mode='max', factor=0.5, patience=3)  # 当评价指标不在提升时，减少学习率
# checkpoint = ModelCheckpoint('./model/best_0105.hdf5', monitor='val_tag_crf_Sparse_accuracy', verbose=2, save_best_only=True, mode='max',
#                                      save_weights_only=True)  # 保存最好的模型


model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator),
        callbacks=[early_stopping, plateau, evaluator],
    )

