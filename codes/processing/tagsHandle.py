import random
import os
import pickle
from setting import ADD_SAMPLE_PATH, PATH, DL_DATA_FIRST, STRENGTH_DATA_FILE_NAME


'''
    对地址词汇和标签进行处理，使其转变为神经网络要求的输入格式
'''


def readDataFiles(path):
    '''
    Purpose:                逐行读取文件，将每一行看作是一个字符串存放在一个list中，shuffle这个list之后返回
    Input:
        path:               string，文件所在地址
    Output:
        cat_sample_20411:   list，所有行字符串集合
    '''
    with open(path,'r', encoding='utf-8')as g_cat:
        all_data = []
        for items in g_cat:
            item = items.strip().replace(' ','#')
            words = item.split('|')[0].split(',')
            labels = item.split('|')[1].split(',')
            all_data.append((words, labels))
    random.shuffle(all_data)
    return all_data


def labelExtend(words, labels):
    words_char = []
    labels_char = []
    for i,word in enumerate(words):
        middle = False
        if labels[i] not in ['R1', 'R2', 'R3', 'R4']:
            middle = True
        if len(word)==1:
            word_char=[word]
            label_char=[labels[i]]
        elif len(word)==2:
            word_char=[word[0],word[-1]]
            label_char= ['M', labels[i]] if middle else ['X',labels[i]]
        elif len(word)>2:
            word_char=[word[s] for s in range(len(word))]
            label_char= ['M'] * len(word) if middle else ['X'] * len(word)
            label_char[-1]=labels[i]
        words_char.extend(word_char)
        labels_char.extend(label_char)
    return words_char, labels_char


def labelExtend_only_x(words, labels):
    words_char = []
    labels_char = []
    for i,word in enumerate(words):
        word = word.strip().replace(' ','#')
        if len(word)==1:
            word_char=[word]
            label_char=[labels[i]]
        elif len(word)==2:
            word_char=[word[0],word[-1]]
            label_char= ['X',labels[i]]
        elif len(word)>2:
            word_char=[word[s] for s in range(len(word))]
            label_char= ['X'] * len(word)
            label_char[-1]=labels[i]
        # 验证label是否正确
        right_tags = ['X', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R30', 'R31', 'R90', 'R99']
        for lc in label_char:
            if lc not in right_tags:
                print((words, labels))
                raise ValueError('label error')
        words_char.extend(word_char)
        labels_char.extend(label_char)
    return words_char, labels_char


def shuffle_data(words, tags):
    zip2list = [j for j in zip(words, tags)]
    # 乱序
    random.shuffle(zip2list)
    words_dell, tags_dell = zip(*zip2list)
    words_dell, tags_dell = list(words_dell), list(tags_dell)
    return words_dell, tags_dell


def readPKlFiles(path):
    readList = []
    with open(path, 'rb') as f:
        readList = pickle.load(f)
    return readList


def tagMain():
    strength_data = readDataFiles('data/generate/strength/first_second_v0129.txt')
    dl_data = readPKlFiles('data/source/r20_r22_r23_r24_twoR1-R4clean/train_all_data.pkl')
    
    dl_data.extend(strength_data)

    print(len(dl_data))
    random.shuffle(dl_data)
    validation_num = int(len(dl_data)*0.2)
    with open(PATH+'generate/train_data/data0129_bidata/dev0129_all.txt','w',encoding='utf-8') as dev:
        for items in dl_data[:validation_num]:
            words, labels = items
            if len(words) != len(labels) or len(words) == 0 or words[0] == '':
                continue
            words_char, labels_char = labelExtend_only_x(words, labels)
            # words_char_reversed, labels_char_reversed = labelExtend_only_x(words[::-1], labels[::-1])
            for t,char in enumerate(words_char):
                dev.write(char+' '+labels_char[t]+'\n')
            dev.write('\n')

    with open(PATH+'generate/train_data/data0129_bidata/train0129_all.txt','w',encoding='utf-8') as train:
        for items in dl_data[validation_num:]:
            words, labels = items
            if len(words) != len(labels) or len(words) == 0 or words[0] == '':
                continue
            words_char, labels_char = labelExtend_only_x(words, labels)
            words_char_reversed, labels_char_reversed = labelExtend_only_x(words[::-1], labels[::-1])
            for p,char in enumerate(words_char):
                train.write(char+' '+labels_char[p]+'\n')
            train.write('\n')
            for p,char in enumerate(words_char_reversed):
                train.write(char+' '+labels_char_reversed[p]+'\n')
            train.write('\n')