import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from hanlp.components.pos import RNNPartOfSpeechTagger

tagger = RNNPartOfSpeechTagger()
save_moedl='./model/pos/'
tagger.load(save_moedl)
address=[list('广东省揭阳市揭西县良田乡下村村委田坑村30号')]
' '.join(tagger.predict(address)[0])