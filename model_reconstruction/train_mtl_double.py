import os
import pickle
from tqdm import tqdm
PATH = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".") + '/data/'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
# from bert4keras.layers import ConditionalRandomField
from custom_crf import ConditionalRandomField
from keras.layers import *
from keras.models import Model
from keras.layers import Lambda
from keras.callbacks import *
# from keras import backend as K
from tqdm import tqdm

from hanlp.optimizers.adamw.optimization import AdamWeightDecay


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








def process_data(file_path):
    D=[]
    with open(file_path,encoding='utf-8') as f:
        f = f.read()
        for l in tqdm(f.split('\n\n')):
            add_list=l.split('\n')
            if len(add_list)!=0:
                add,tag,temp=[],[],''
                for item in add_list:
                    if len(item.split(' '))==2:
                        k,v=item.split(' ')
                        temp=temp+k
                    if v!='X':
                        add.append(temp)
                        tag.append(v)
                        temp=''
                D.append((add,tag))
    return D

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
                tag_labels+=([label2id['X']]*(len(w_token_ids)-1)+[label2id[tag]])
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
                yield [batch_token_ids,batch_seq_labels,batch_tag_labels], [batch_seq_labels,batch_tag_labels]
                batch_token_ids, batch_seq_labels, batch_tag_labels = [], [], []



    
def build_optimizer(optimizer):
    if isinstance(optimizer, (str, dict)):
        custom_objects = {'AdamWeightDecay': AdamWeightDecay(learning_rate=0.0001,
                                                                 weight_decay_rate=0.01,
                                                                 beta_1=0.9,
                                                                 beta_2=0.999,
                                                                 epsilon=1e-6,
                                                                 exclude_from_weight_decay=['layer_norm', 'bias'])}
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.utils.deserialize_keras_object(optimizer,
                                                                                           module_objects=tf.keras.optimizers,
                                                                                           custom_objects=custom_objects)
    return optimizer

def build_model(embeddings=100,vocab_size=vocab_size,rnn_units=100):
   
    x_in = Input(shape=(None,))
    output=Embedding(input_dim=vocab_size, output_dim=embeddings,
                                            trainable=True)(x_in)
    
    output=Masking(mask_value=0)(output)
    output=Bidirectional(LSTM(units=300, return_sequences=True),name='bilstm_1')(output)
    
    seq_output=Dropout(0.5)(output)
    
    
    output=Bidirectional(LSTM(units=100, return_sequences=True),name='bilstm_2')(output)
    
    tag_output=Dropout(0.5)(output)
    
    seq_output=TimeDistributed(Dense(num_seglabels), name='dense_seq')(seq_output)
    
    tag_output=TimeDistributed(Dense(num_labels), name='dense_tag')(tag_output)
    
#     CRF=ConditionalRandomField(seq_lr_multiplier=seq_crf_lr_multiplier,tag_lr_multiplier=tag_crf_lr_multiplier,name='crf')
    
#     crf_out=CRF([seq_output,tag_output])

    model = Model(x_in, [seq_output,tag_output])
#     model.summary()

#     model.compile(
# #         loss=None,
#         loss=[CRF.seq_sparse_loss,CRF.tag_sparse_loss],
# #         optimizer=Adam(learing_rate),
#         optimizer=build_optimizer('Adam'),
#         metrics=[CRF.sparse_accuracy]
#     )
    return model



def get_model(embeddings=100,vocab_size=vocab_size,rnn_units=100):
    
    model_=build_model(embeddings=embeddings,vocab_size=vocab_size,rnn_units=rnn_units)
    seq_output,tag_output=model_.output
    seq_true=Input(shape=(None,))
    tag_true=Input(shape=(None,))
    CRF=ConditionalRandomField(seq_lr_multiplier=seq_crf_lr_multiplier,tag_lr_multiplier=tag_crf_lr_multiplier,name='crf')
    
    crf_out=CRF([seq_output,tag_output,seq_true,tag_true])
#     model_ouput=CustomMultiLossLayer(crf_list=[Seq_crf,Tag_crf],nb_outputs=2,name='crf_loss')([[seq_true,tag_true],[seq_output,tag_output]])
    
    model = Model([model_.inputs,seq_true,tag_true], crf_out)
    
    model.summary()
    
    model.compile(
        loss=None,
        optimizer=build_optimizer('Adam'),
        metrics=[CRF.sparse_accuracy]
    )
    
    return model,CRF


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text):
        pass
    
    


# train_path=PATH+'generate/pkl/train_all.pkl'
# train_file=pickle.load(open(train_path,'rb'))
# np.random.shuffle(train_file)
# val_len=int(0.8*len(train_file))
# train_generator = data_generator(train_file[:val_len], batch_size=batch_size)
# valid_generator=data_generator(train_file[val_len:],batch_size=batch_size)


train_,valid_=process_data(PATH+'generate/train0121_all.txt'),process_data(PATH+'generate/dev0121_all.txt')
train_generator = data_generator(train_, batch_size=batch_size)
valid_generator=data_generator(valid_,batch_size=batch_size)



model,CRF = get_model(embeddings=200,vocab_size=vocab_size,rnn_units=300)


# evaluator = Evaluator(valid_data,model,Seq_ner,Tag_ner)
early_stopping = EarlyStopping(monitor='val_crf_1_sparse_accuracy', patience=25)  # 早停法，防止过拟合
plateau = ReduceLROnPlateau(monitor='val_crf_1_sparse_accuracy', verbose=1, mode='max', factor=0.5, patience=3)  # 当评价指标不在提升时，减少学习率
checkpoint = ModelCheckpoint('./model/best_mtl_double_0126_1.h5', monitor='val_crf_1_sparse_accuracy', verbose=2, save_best_only=True, mode='max',
                                     save_weights_only=True)  # 保存最好的模型


model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator),
        callbacks=[early_stopping, plateau,checkpoint],
    )

