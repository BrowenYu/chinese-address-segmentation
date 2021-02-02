import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from hanlp.components.pos import RNNPartOfSpeechTagger
tagger=RNNPartOfSpeechTagger()
save_moedl='./model/pos110202'
PATH = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".") + '/data/'
tagger.fit(PATH+'/generate/train1102.txt',
           PATH+'/generate/dev1102.txt',
           save_dir=save_moedl,
           epochs=2000,
           metrics='accuracy',
           embeddings=100,
           optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
           batch_size=512,
           embedding_trainable=True,
           verbose=True,
           rnn_input_dropout=0.5,
           rnn_units=300,
           rnn_output_dropout=0.5
           )
