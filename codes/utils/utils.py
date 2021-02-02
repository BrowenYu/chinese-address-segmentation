import pickle
import random
from pyspark.sql import SparkSession


def txt2pklForLabel(txtAddr, pklAddr, label, mode='r', encoding='utf-8'):
    '''
    将txt文件转化为带标签的pkl文件
    '''
    f = open(txtAddr, mode=mode, encoding=encoding)
    file_line_list = []
    for item in f:
        items = item.strip()
        file_line_list.append([items, label])
    f.close()
    pk = open(pklAddr, 'wb')
    pickle.dump(file_line_list, pk)
    pk.close()


def substractArea(readPath, writePath, level, dic_vcab, labelAddr, nameAddr, shortAddr=-1):
    '''
    提取文件中的地址词汇并添加其对应的等级
    '''
    f = open(readPath, 'r', encoding='utf-8')
    g = open(writePath, 'wb')
    for items in f:
        item = items.strip().split(',')
        name = item[nameAddr]
        label = item[labelAddr]
        if shortAddr >= 0:
            s = item[shortAddr]
            if label == level:
                g.write(name+'|R'+str(level)+'\n')
                if shortAddr.strip() != '""':
                    g.write(name+'|R'+str(level)+'\n')
                    if name not in dic_vcab:dic_vcab[name] = s
    f.close()
    g.close()


def shuffleList(rdAddr, wtAddr, shuffleCount=3, maxLine=1000000):
    '''
    随机打乱数据序列
    '''
    f=open(rdAddr,'r', encoding='utf-8')
    sample_ls=[]
    for items in f:
        item=items.strip()
        sample_ls.append(item)
    f.close()
    for i in range(shuffleCount):   
        random.shuffle(sample_ls)
    #写入文件
    g=open(wtAddr,'w', encoding='utf-8')
    for lines in sample_ls[:maxLine]:
        g.write(lines+'\n')
    g.close()


def readPkl(rdAddr):
    '''
    加载pkl文件，返回列表
    '''
    readList = []
    with open(rdAddr, 'rb') as f:
        readList=pickle.load(f)
    random.shuffle(readList)
    print(rdAddr + " has loaded!")
    return readList


def loadCSVFiles(addrList, dataStr):
    '''
    加载pkl文件，返回列表
    '''
    spark=SparkSession.builder.master("local").getOrCreate()
    for i in range(len(addrList)):
        df=spark.read.parquet("hdfs://10.100.2.50:8020"+addrList[i])
        df.repartition(1).write.mode("Overwrite").csv("./cat_over"+str(i+1)+"_v"+dataStr+".csv",header=False)


def getLevelWordsFromCSV(level, rdAddr):
    '''
    从cn_division_new.csv中获取特定级别的数据，其中第一个参数对应等级，第二个参数对应文件所在地址
    注意：如果文件中地址词汇、标签的位置发生改变，需要改变一些数值，在下面以注释的形式给出
    '''
    levelWords = []
    f = open(rdAddr, encoding='utf-8')
    for item in f:
        item=item.strip().split(',')
        word = eval(item[6])    # 如果文件中地址词汇的位置发生改变，需要改变这里的数值
        tag = eval(item[5])     # 如果文件中地址标签的位置发生改变，需要改变这里的数值
        if tag == str(level):
            levelWords.append([word, 'R'+str(level)])
    f.close()
    return levelWords


def getLevelWordsFromTXT(level, rdAddr):
    '''
    dl_data_v4_21103.txt中获取特定级别的数据，其中第一个参数对应等级，第二个参数对应文件所在地址
    '''
    levelWords = []
    f = open(rdAddr, encoding='utf-8')
    for item in f:
        item=item.strip().split(',')
        for i in range(len(item)):
            if i % 2 == 1 and item[i] == 'R'+str(level):
                levelWords.append(item[i-1])
    f.close()
    return levelWords


def getSplitWords(addr, tag):
    '''
    利用model预测结果tag对地址addr划分标注，将addr转化为“北京市,R1,朝阳区,R2,知春路,R3”这样的形式
    addr为原始地址，tag为模型预测的结果
    '''
    strList = []
    tagCur = ''
    for i in range(len(tag)):
        if tag[i] != 'X' and tag[i] != 'M':
            if i != 0:
                strList.append(',')
                strList.append(tagCur)
                strList.append(',')
            strList.append(addr[i])
            tagCur = tag[i]
        else:
            strList.append(addr[i])
    strList.append(',')
    strList.append(tagCur)
    return ''.join(strList)


# 加强数据模块数据处理工具
def deleteRandom(word, tag):
    if random.uniform(0, 1) <= 0.3 and len(tag) >= 3:
        del word[0]
        del tag[0]
        if random.uniform(0, 1) <= 0.5:
            ix = list(range(len(tag)))
            random.shuffle(ix)
            del word[ix[0]]
            del tag[ix[0]]
        elif random.uniform(0, 1) <= 0.5:
            ix = list(range(len(tag)))
            random.shuffle(ix)
            del word[ix[-1]]
            del tag[ix[-1]]
        else:
            pass


def shuffle_first(words,tags):
    zip2list = [j for j in zip(words, tags)]
    # 乱序
    random.shuffle(zip2list)
    words_dell, tags_dell = zip(*zip2list)
    words_dell, tags_dell = list(words_dell), list(tags_dell)
    # 乱序+删除
    if random.uniform(0, 1) <= 0.5:
        words_dell,tags_dell = del_first(words_dell,tags_dell)
    return words_dell,tags_dell


def del_first(words,tags):
    delNum = random.randint(1, len(words))
    delIndex = random.sample(range(0, len(words)), delNum)
    words_new = []
    tags_new = []
    for j in range(len(words)):
        if j not in delIndex:
            words_new.append(words[j])
            tags_new.append(tags[j])
    return words_new,tags_new