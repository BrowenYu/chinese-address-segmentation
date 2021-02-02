import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import tensorflow.keras.backend as k
import tensorflow.keras as keras
from hanlp.transform.tsv import TSVTaggingTransform
from hanlp.components.pos import RNNPartOfSpeechTagger
from tqdm import tqdm

from model_main import build_model, load_vocabs, build


save_dir = "save_dir/"
model_dir = "model_dir/0129_best_model/"

def load_model():
    transform = TSVTaggingTransform()
    # 读取字典
    load_vocabs(transform, save_dir)
    # # 构建模型
    # model=build(transform)
    # # 加载参数
    # model.load_weights(model_dir + "model.h5")
    # return model, transform
    return transform


def predict(pre_str_list, transform, model):
    pre_chars = []
    for s in pre_str_list:
        pre_chars.append(list(s))
#     pre_chars = [list(pre_str)]
    data, flat = transform.input_to_inputs(pre_chars)
    dataset = transform.inputs_to_dataset(data, batch_size=512)
    pre_tags = []
    for idx, batch in enumerate(dataset):
        X = batch[0]
        Y = model.predict_on_batch(batch)
        for output in transform.Y_to_outputs(Y, X=X, inputs=data):
            pre_tags.extend(output)
    return pre_tags


def evaluate_f1(data, model, transform, batch_size):
    label = ['R1','R2','R3','R4','R5','R6','R7','R20','R21','R22','R23','R24','R25','R30','R31','R90','R99']
    f1_dict = {item: 0 for item in label}
    la_idx = transform.y_to_idx(tf.constant(label)).numpy()
    # print(la_idx)
    pnum_dict = {item: 1e-10 for item in la_idx}
    rnum_dict = {item: 1e-10 for item in la_idx}
    pall_dict = {item: 1e-10 for item in la_idx}
    rall_dict = {item: 1e-10 for item in la_idx}
    viterbi_sequence = []
    true_label = []
    for x,y in tqdm(data):
        d = x.numpy()
        temp = model.predict(d, batch_size=batch_size)
        temp = tf.argmax(temp, axis=2)
        viterbi_sequence.extend([i for i in temp.numpy()])
        true_label.extend(y.numpy())
    for true, pred in zip(true_label,viterbi_sequence):
        for i in range(np.size(true,0)):
            # 统计单个分类召回
            if true[i] in la_idx:
                if true[i]==pred[i]:
                    right = True
                    j = i
                    while(j<len(true) and true[j] not in la_idx):
                        if pred[j] in la_idx:
                            right = False
                        j += 1
                    if pred[j] not in la_idx:
                        right =False
                    rnum_dict[true[i]] += right
                rall_dict[true[i]] += 1
            # 统计单个分类准确
            if pred[i] in la_idx:
                if true[i] == pred[i]:
                    right = True
                    j = i
                    while(j<len(pred) and pred[j] not in la_idx):
                        if true[j] in la_idx:
                            right = False
                        j += 1
                    if true[j] not in la_idx:
                        right =False
                    pnum_dict[true[i]] += right
                pall_dict[pred[i]] += 1
                
    for k in f1_dict.keys():
        idx = transform.y_to_idx(tf.constant([k])).numpy()[0]
        # 单个分类准确率
        p = pnum_dict[idx]/pall_dict[idx]
        # 单个分类召回率
        r = rnum_dict[idx]/rall_dict[idx]
        f1_dict[k] = [p, r, 2*(p*r)/(p+r)]
    print('Acc, Recall, F1: {}'.format(f1_dict))
    return f1_dict

if __name__ == "__main__":
    pre_str = "广东"
    # pre_str = ["广", "东", "<pad>"]
    transform = load_model()
    # model.save('model_dir/my_model_')
    new_model = tf.keras.models.load_model(model_dir,compile=False)
    # output = predict([pre_str], transform, new_model)
    # print(output)

    dev_path = '/home/bangsun/research/fank/best_model_code/model_reconstruction/dataset/data0121_bidata/dev0121_all.txt'
    batch_size = 512
    dev_data = transform.file_to_dataset(dev_path, batch_size=batch_size, shuffle=True, repeat=None)
    f1_dict = evaluate_f1(dev_data, new_model, transform, batch_size)