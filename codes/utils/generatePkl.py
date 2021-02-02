import pickle
import random

'''
    生成特定等级词汇
'''

'''
    Purpose:从entity_R99.txt生成R99等级的词汇
    Input:
        rdPath:     指定读取文件的地址
        wtPath:     指定写入pkl文件的地址
'''
def generateR99(rdPath, wtPath):
    f = open(rdPath, mode='r', encoding='utf-8')
    entity_list_r99=[]
    for items in f:
        line=items.strip().split(',')
        sentences_tags=line[:len(line)-1]
        if len(sentences_tags) % 2 == 0 :
            word=[]
            tag=[]
            for i in range(len(sentences_tags)):
                if i % 2 == 0 and len(sentences_tags[i].strip())>0  and  sentences_tags[i+1].strip() == 'R99':
                    entity_list_r99.append([sentences_tags[i],sentences_tags[i+1]])
    pk=open(wtPath, 'wb')
    random.shuffle(entity_list_r99)
    pickle.dump(entity_list_r99,pk)
    pk.close()

'''
    Purpose:从原始数据中获取特定等级的词汇
    Input:
        rdPath:     指定读取文件的地址
        wtPath:     指定写入pkl文件的地址
        pickList:   指定的等级列表(类型必须是列表)
'''
def generateFromOriginal(rdPath, wtPath, pickList):
    f=open(rdPath , mode='r',encoding='gbk')
    entity_list=[]
    for items in f:
        line=eval(items.strip()).split(',')
        sentences_tags=line[:len(line)-1]
        if len(sentences_tags) % 2 == 0 :
            for i in range(len(sentences_tags)):
                if i % 2 == 0 and len(sentences_tags[i].strip())>0:
                    if sentences_tags[i+1].strip() in pickList:
                        entity_list.append([sentences_tags[i],sentences_tags[i+1]])
    pk=open(wtPath, 'wb')
    random.shuffle(entity_list)
    pickle.dump(entity_list,pk)
    pk.close()

if __name__ == '__main__':
    a = "E:\\bangsun/code/codes/utils/test.pkl"
    generateFromOriginal('E:\\bangsun/code/data/source/dl_data_v4_20805.txt',a, ['R30', 'R20', 'R31', 'R24', 'R21', 'R25', 'R22', 'R23'])