import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from hanlp.components.pos import RNNPartOfSpeechTagger
tagger = RNNPartOfSpeechTagger()
save_moedl='./model/pos1019/'
tagger.load(save_moedl)
import time
tm=time.strftime('%Y%m%d%H%M%S')
save_moedl='./model/model_serving/serving1019{}'.format(tm)
tagger.export_model_for_serving(export_dir=save_moedl,version=1,overwrite=True)