import numpy as np
from utils.utils import txt2pklForLabel, substractArea, readPkl, shuffleList
import random
import pickle


class savePkl():
    def __init__(self, PATH):
        self.r99_receiver_0906 = readPkl(PATH + 'generate/pkl/' + 'r99_receiver_0906.pkl')
        self.r90_direction_0906 = readPkl(PATH + 'generate/pkl/' + 'r90_direction_0906.pkl')
        self.r0_entityAddr_0910 = readPkl(PATH + 'generate/pkl/' + 'r0_location_1016.pkl')
        self.r24_addvillage_0930 = readPkl(PATH + 'generate/pkl/' + 'r24_addvillage_0930.pkl')
        self.r0_intersection_0906 = readPkl(PATH + 'generate/pkl/' + 'r0_intersection_0906.pkl')
        self.r0_roadDir_1014 = readPkl(PATH + 'generate/pkl/' + 'r0_roadDir_1014.pkl')
        self.r0_cityBuilding_1015 = readPkl(PATH + 'generate/pkl/' + 'r0_cityBuilding_1015.pkl')
        self.r24_StreetOffice_1014 = readPkl(PATH + 'generate/pkl/' + 'r24_StreetOffice_1014.pkl')
        self.r5_road_1014 = readPkl(PATH + 'generate/pkl/' + 'r5_road_1014.pkl')
        self.r6_roadNum_0906 = readPkl(PATH + 'generate/pkl/' + 'r6_roadNum_0906.pkl')
        self.r1_Province_1026 = readPkl(PATH + 'generate/pkl/' + 'r1_Province_1026.pkl')
        self.r2_city_1026 = readPkl(PATH + 'generate/pkl/' + 'r2_city_1026.pkl')
        self.r3_town_1026 = readPkl(PATH + 'generate/pkl/' + 'r3_town_1026.pkl')
        self.r4_country_1026 = readPkl(PATH + 'generate/pkl/' + 'r4_country_1026.pkl')
        r4_country_1026 = readPkl(PATH + 'generate/pkl/' + 'r4_country_1026.pkl')
        self.zhen = []
        self.xiang = []
        self.cun = readPkl(PATH + 'generate/pkl/' + 'r5_village2_1015.pkl')
        self.num_list = createNumList()
        for item in r4_country_1026:
            if item[0][-1] == '镇':
                self.zhen.append(item[0])
            if item[0][-1] == '乡':
                self.xiang.append(item[0])

def case1(sp):
    if random.uniform(0, 1) <= 0.5:
        road = random.sample(sp.r0_roadDir_1014, 1)[0]
    else:
        road = list(random.sample(sp.r0_intersection_0906, 1)[0])
        direction = random.sample(sp.r90_direction_0906, 1)[0]
        if direction[0] in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
            road[0] = [direction[0]] + road[0]
            road[1] = [direction[1]] + road[1]
        else:
            road[0] = road[0] + [direction[0]]
            road[1] = road[1] + [direction[1]]
    entityAddr = random.sample(sp.r0_entityAddr_0910, 1)[0]
    cityBuilding = random.sample(sp.r0_cityBuilding_1015, 1)[0]
    words = road[0] + entityAddr[0] + cityBuilding[0]
    tags = road[1] + entityAddr[1] + cityBuilding[1]
    return words, tags


def case2(sp):
    if random.uniform(0, 1) <= 0.4:
        cityBuilding = random.sample(sp.r0_cityBuilding_1015, 1)[0]
        direction = random.sample(sp.r90_direction_0906, 1)[0]
        if direction[0] in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
            words = [direction[0]] + cityBuilding[0]
            tags = [direction[1]] + cityBuilding[1]
        else:
            words = cityBuilding[0] + [direction[0]]
            tags = cityBuilding[1] + [direction[1]]
    else:
        if random.uniform(0, 1) <= 0.4:
            entityAddr = random.sample(sp.r0_entityAddr_0910, 1)[0]
            words = entityAddr[0]
            tags = entityAddr[1]
        else:
            cityBuilding = random.sample(sp.r0_cityBuilding_1015, 1)[0]
            words = cityBuilding[0]
            tags = cityBuilding[1]
    return words, tags


def case3(sp):
    villageOffice = random.sample(sp.r24_addvillage_0930, 6000)
    StreetOffice = random.sample(sp.r24_StreetOffice_1014 + villageOffice, 1)[0]
    if random.uniform(0, 1) <= 0.6:
        if random.uniform(0, 1) <= 0.4:
            words, tags = case2(sp)
        else:
            words, tags = case1(sp)
        words = StreetOffice[0] + words
        tags = StreetOffice[1] + tags
    elif random.uniform(0, 1) <= 0.2:
        if random.uniform(0, 1) <= 0.5:
            road = random.sample(sp.r0_roadDir_1014, 1)[0]
        else:
            road = list(random.sample(sp.r0_intersection_0906, 1)[0])
            direction = random.sample(sp.r90_direction_0906, 1)[0]
            if direction[0] in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
                road[0] = [direction[0]] + road[0]
                road[1] = [direction[1]] + road[1]
            else:
                road[0] = road[0] + [direction[0]]
                road[1] = road[1] + [direction[1]]
        words = road[0] + StreetOffice[0]
        tags = road[1] + StreetOffice[1]
    else:
        if random.uniform(0, 1) <= 0.4:
            direction = random.sample(sp.r90_direction_0906, 1)[0]
            if direction[0] in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
                words = [direction[0]] + StreetOffice[0]
                tags = [direction[1]] + StreetOffice[1]
            else:
                words = StreetOffice[0] + [direction[0]]
                tags = StreetOffice[1] + [direction[1]]
        elif random.uniform(0, 1) <= 0.6:
            words = StreetOffice[0]
            tags = StreetOffice[1]
        elif random.uniform(0, 1) <= 0.5:
            road = random.sample(sp.r5_road_1014, 1)[0]
            words = StreetOffice[0] + [road[0]]
            tags = StreetOffice[1] + [road[1]]
        else:
            cun = random.sample(sp.cun, 1)[0]
            words = StreetOffice[0] + [cun[0]]
            tags = StreetOffice[1] + [cun[1]]
    return words, tags


def case4(sp):
    if random.uniform(0, 1) <= 0.5:
        road = random.sample(sp.r0_roadDir_1014, 1)[0]
    else:
        road = list(random.sample(sp.r0_intersection_0906, 1)[0])
    roadNum = random.sample(sp.r6_roadNum_0906, 1)[0]
    entityAddr = random.sample(sp.r0_entityAddr_0910, 1)[0]
    cityBuilding = random.sample(sp.r0_cityBuilding_1015, 2)
    direction = random.sample(sp.r90_direction_0906, 1)[0]
    if direction[0] in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
        if random.uniform(0, 1) <= 0.5:
            words = road[0] + [roadNum[0]] + [direction[0]] + entityAddr[0] + cityBuilding[0][0]
            tags = road[1] + [roadNum[1]] + [direction[1]] + entityAddr[1] + cityBuilding[0][1]
        else:
            words = road[0] + [roadNum[0]] + [direction[0]] + cityBuilding[0][0] + cityBuilding[1][0]
            tags = road[1] + [roadNum[1]] + [direction[1]] + cityBuilding[0][1] + cityBuilding[1][1]
    else:
        if random.uniform(0, 1) <= 0.5:
            words = road[0] + [roadNum[0]] + entityAddr[0] + [direction[0]] + cityBuilding[0][0]
            tags = road[1] + [roadNum[1]] + entityAddr[1] + [direction[1]] + cityBuilding[0][1]
        else:
            words = road[0] + [roadNum[0]] + cityBuilding[0][0] + [direction[0]] + cityBuilding[0][0]
            tags = road[1] + [roadNum[1]] + cityBuilding[0][1] + [direction[1]] + cityBuilding[0][1]
    return words, tags


def createNum(num):
    p = []
    for i in range(num):
        p.append(random.randint(1, 1000))
    s = str(p[0])
    for i in range(num - 1):
        s = s + '-' + str(p[i + 1])
    return s


def createNumList():
    numList = []
    for i in range(1000):
        num = ([createNum(2)], ["R6"])
        numList.append(num)
        num = ([createNum(3)], ["R6"])
        numList.append(num)
        num = ([createNum(4)], ["R6"])
        numList.append(num)
    return numList


"""
    Purpose:            添加road+XX-XX-XX形式数据
"""


def case5(sp):
    road = random.sample(sp.r5_road_1014, 1)[0]
    roadNum = random.sample(sp.num_list, 1)[0]
    direction = random.sample(sp.r90_direction_0906, 1)[0]
    cityBuilding = random.sample(sp.r0_cityBuilding_1015, 1)[0]
    if direction[0] in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
        if random.uniform(0, 1) <= 0.5:
            words = [road[0]] + roadNum[0] + [direction[0]] + cityBuilding[0]
            tags = [road[1]] + roadNum[1] + [direction[1]] + cityBuilding[1]
        else:
            words = [road[0]] + roadNum[0] + [direction[0]] + cityBuilding[0] + cityBuilding[0]
            tags = [road[1]] + roadNum[1] + [direction[1]] + cityBuilding[1] + cityBuilding[1]
    elif random.uniform(0, 1) <= 0.5:
        if random.uniform(0, 1) <= 0.5:
            words = [road[0]] + roadNum[0] + [direction[0]] + cityBuilding[0]
            tags = [road[1]] + roadNum[1] + [direction[1]] + cityBuilding[1]
        else:
            words = [road[0]] + roadNum[0] + cityBuilding[0] + [direction[0]] + cityBuilding[0]
            tags = [road[1]] + roadNum[1] + cityBuilding[1] + [direction[1]] + cityBuilding[1]
    else:
        words = [road[0]] + roadNum[0]
        tags = [road[1]] + roadNum[1]
    return words, tags


"""
    Purpose:            两个路连续出现
"""


def case6(sp):
    road1 = random.sample(sp.r5_road_1014, 1)[0]
    road2 = random.sample(sp.r5_road_1014, 1)[0]
    if random.uniform(0, 1) <= 0.4:
        cityBuilding = random.sample(sp.r0_cityBuilding_1015, 1)[0]
        direction = random.sample(sp.r90_direction_0906, 1)[0]
        if direction[0] in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
            words = [direction[0]] + [road1[0]] + [road2[0]] + cityBuilding[0]
            tags = ['R90'] + [road1[1]] + [road2[1]] + cityBuilding[1]
        else:
            words = [road1[0]] + [road2[0]] + cityBuilding[0] + [direction[0]]
            tags = [road1[1]] + [road2[1]] + cityBuilding[1] + [direction[1]]
    else:
        if random.uniform(0, 1) <= 0.4:
            entityAddr = random.sample(sp.r0_entityAddr_0910, 1)[0]
            words = [road1[0]] + [road2[0]] + entityAddr[0]
            tags = [road1[1]] + [road2[1]] + entityAddr[1]
        else:
            cityBuilding = random.sample(sp.r0_cityBuilding_1015, 1)[0]
            words = [road1[0]] + [road2[0]] + cityBuilding[0]
            tags = [road1[1]] + [road2[1]] + cityBuilding[1]
    return words, tags


"""
    Purpose:            行政区划直接加R99
"""


def case7(sp):
    r99 = random.sample(sp.r99_receiver_0906, 1)[0]
    words = [r99[0]]
    tags = [r99[1]]
    return words, tags


"""
        Purpose:            两个方向词同时出现
"""


def case8(sp):
    if random.uniform(0, 1) <= 0.5:
        road = random.sample(sp.r0_roadDir_1014, 1)[0]
    else:
        road = list(random.sample(sp.r0_intersection_0906, 1)[0])
    roadNum = random.sample(sp.r6_roadNum_0906, 1)[0]
    entityAddr = random.sample(sp.r0_entityAddr_0910, 1)[0]
    cityBuilding = random.sample(sp.r0_cityBuilding_1015, 1)[0]
    direction = random.sample(sp.r90_direction_0906, 1)[0]
    if direction[0] in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
        if random.uniform(0, 1) <= 0.5:
            words = road[0] + [roadNum[0]] + [direction[0]] + entityAddr[0] + cityBuilding[0]
            tags = road[1] + [roadNum[1]] + [direction[1]] + entityAddr[1] + cityBuilding[1]
        else:
            words = road[0] + [roadNum[0]] + [direction[0]] + cityBuilding[0] + cityBuilding[0]
            tags = road[1] + [roadNum[1]] + [direction[1]] + cityBuilding[1] + cityBuilding[1]
    else:
        direction1 = random.sample(sp.r90_direction_0906, 1)[0]
        if random.uniform(0, 1) <= 0.5:
            if direction1[0] not in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
                words = road[0] + [roadNum[0]] + entityAddr[0] + [direction[0]] + [direction1[0]] + cityBuilding[0]
                tags = road[1] + [roadNum[1]] + entityAddr[1] + [direction[1]] + [direction1[1]] + cityBuilding[1]
            else:
                words = road[0] + [roadNum[0]] + entityAddr[0] + [direction[0]] + cityBuilding[0]
                tags = road[1] + [roadNum[1]] + entityAddr[1] + [direction[1]] + cityBuilding[1]
        else:
            if direction1[0] not in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
                words = road[0] + [roadNum[0]] + cityBuilding[0] + [direction[0]] + [direction1[0]] + cityBuilding[0]
                tags = road[1] + [roadNum[1]] + cityBuilding[1] + [direction[1]] + [direction1[1]] + cityBuilding[1]
            else:
                words = road[0] + [roadNum[0]] + cityBuilding[0] + [direction[0]] + cityBuilding[0]
                tags = road[1] + [roadNum[1]] + cityBuilding[1] + [direction[1]] + cityBuilding[1]
    return words, tags


'''
        Purpose:            方位词：'内'
'''


def case9(sp):
    if random.uniform(0, 1) <= 0.5:
        road = random.sample(sp.r0_roadDir_1014, 1)[0]
    else:
        road = list(random.sample(sp.r0_intersection_0906, 1)[0])
    roadNum = random.sample(sp.r6_roadNum_0906, 1)[0]
    entityAddr = random.sample(sp.r0_entityAddr_0910, 1)[0]
    cityBuilding = random.sample(sp.r0_cityBuilding_1015, 1)[0]
    direction = (['内'], ['R90'])
    if random.uniform(0, 1) <= 0.5:
        words = road[0] + [roadNum[0]] + entityAddr[0] + direction[0] + cityBuilding[0]
        tags = road[1] + [roadNum[1]] + entityAddr[1] + direction[1] + cityBuilding[1]
    else:
        words = road[0] + [roadNum[0]] + cityBuilding[0] + direction[0]
        tags = road[1] + [roadNum[1]] + cityBuilding[1] + direction[1]
    return words, tags


'''
        Purpose:            最后一个地标加方位词
'''


def case10(sp):
    if random.uniform(0, 1) <= 0.5:
        road = random.sample(sp.r0_roadDir_1014, 1)[0]
    else:
        road = list(random.sample(sp.r0_intersection_0906, 1)[0])
    roadNum = random.sample(sp.r6_roadNum_0906, 1)[0]
    entityAddr = random.sample(sp.r0_entityAddr_0910, 1)[0]
    cityBuilding = random.sample(sp.r0_cityBuilding_1015, 1)[0]
    direction = random.sample(sp.r90_direction_0906, 1)[0]
    if direction[0] in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
        if random.uniform(0, 1) <= 0.5:
            words = road[0] + [roadNum[0]] + [direction[0]] + entityAddr[0] + cityBuilding[0]
            tags = road[1] + [roadNum[1]] + [direction[1]] + entityAddr[1] + cityBuilding[1]
        else:
            words = road[0] + [roadNum[0]] + [direction[0]] + cityBuilding[0] + cityBuilding[0]
            tags = road[1] + [roadNum[1]] + [direction[1]] + cityBuilding[1] + cityBuilding[1]
    else:
        if random.uniform(0, 1) <= 0.5:
            words = road[0] + [roadNum[0]] + entityAddr[0] + cityBuilding[0] + [direction[0]]
            tags = road[1] + [roadNum[1]] + entityAddr[1] + cityBuilding[1] + [direction[1]]
        else:
            words = road[0] + [roadNum[0]] + cityBuilding[0] + cityBuilding[0] + [direction[0]]
            tags = road[1] + [roadNum[1]] + cityBuilding[1] + cityBuilding[1] + [direction[1]]
    return words, tags


'''
        Purpose:            两个地标间不加修饰词
'''


def case11(sp):
    if random.uniform(0, 1) <= 0.5:
        road = random.sample(sp.r0_roadDir_1014, 1)[0]
    else:
        road = list(random.sample(sp.r0_intersection_0906, 1)[0])
    roadNum = random.sample(sp.r6_roadNum_0906, 1)[0]
    entityAddr = random.sample(sp.r0_entityAddr_0910, 1)[0]
    cityBuilding1 = random.sample(sp.r0_cityBuilding_1015, 1)[0]
    cityBuilding2 = random.sample(sp.r0_cityBuilding_1015, 1)[0]
    direction = random.sample(sp.r90_direction_0906, 1)[0]
    if direction[0] in ['靠近', '西邻', '临', '近', '至', '南邻', '毗邻', '东邻', '北邻']:
        if random.uniform(0, 1) <= 0.5:
            words = road[0] + [roadNum[0]] + [direction[0]] + entityAddr[0] + cityBuilding1[0] + cityBuilding2[0]
            tags = road[1] + [roadNum[1]] + [direction[1]] + entityAddr[1] + cityBuilding1[1] + cityBuilding2[1]
        else:
            words = road[0] + [roadNum[0]] + [direction[0]] + cityBuilding1[0] + cityBuilding2[0]
            tags = road[1] + [roadNum[1]] + [direction[1]] + cityBuilding1[1] + cityBuilding2[1]
    else:
        if random.uniform(0, 1) <= 0.5:
            words = road[0] + [roadNum[0]] + entityAddr[0] + [direction[0]] + cityBuilding1[0] + cityBuilding2[0]
            tags = road[1] + [roadNum[1]] + entityAddr[1] + [direction[1]] + cityBuilding1[1] + cityBuilding2[1]
        else:
            words = road[0] + [roadNum[0]] + cityBuilding1[0] + [direction[0]] + cityBuilding2[0]
            tags = road[1] + [roadNum[1]] + cityBuilding1[1] + [direction[1]] + cityBuilding2[1]
    return words, tags


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
    delNum = random.randint(1, 4)
    delIndex = random.sample(range(0, len(words)), delNum)
    words_new = []
    tags_new = []
    for j in range(len(words)):
        if j not in delIndex:
            words_new.append(words[j])
            tags_new.append(tags[j])
    return words_new,tags_new

def process_first(words,tags,p1,p2):
    b = random.uniform(0, 1)
    if b <= p1:
        words, tags = del_first(words, tags)
    elif b <= p2:
        words, tags = shuffle_first(words, tags)
    return words,tags

def getFirstList(sp):
    words_first = []
    tags_first = []
    for x in range(10):
        random.shuffle(sp.r4_country_1026)
        random.shuffle(sp.r3_town_1026)
        random.shuffle(sp.r2_city_1026)
        random.shuffle(sp.r1_Province_1026)
        for i in range(len(sp.r4_country_1026)):
            # 乡镇数据重复
            if random.uniform(0, 1) <= 0.1:
                w = random.sample(sp.r4_country_1026[0], 1)[0][0]
                words = [sp.r1_Province_1026[i % len(sp.r1_Province_1026)][0],
                         sp.r2_city_1026[i % len(sp.r2_city_1026)][0], sp.r3_town_1026[i % len(sp.r3_town_1026)][0],
                         sp.r4_country_1026[i % len(sp.r4_country_1026)][0], w]
                tags = ['R1', 'R2', 'R3', 'R4', 'R4']
            else:
                words = [sp.r1_Province_1026[i % len(sp.r1_Province_1026)][0],
                         sp.r2_city_1026[i % len(sp.r2_city_1026)][0], sp.r3_town_1026[i % len(sp.r3_town_1026)][0],
                         sp.r4_country_1026[i % len(sp.r4_country_1026)][0]]
                tags = ['R1', 'R2', 'R3', 'R4']
            a = random.uniform(0, 1)
            # 整个区域重复三遍
            if a <= 0.3:
                words = words * 3
                tags = tags * 3
                words,tags = process_first(words,tags,0.4,0.8)
            # 整个重复两遍
            elif a < 0.8:
                words = words * 2
                tags = tags * 2
                words,tags = process_first(words,tags,0.4,0.8)
            else:
                words,tags = process_first(words,tags,0.5,1)
            words_first.append(words)
            tags_first.append(tags)
    return words_first, tags_first


def splitAddressUtilR4(cat_v6_over2_20805_shuffle, sp, cat_v6_over2_20805_shuffle_after):
    words_first, tags_first = getFirstList(sp)
    for j in range(len(words_first)):
        i = random.randint(1, 11)
        words_after = []
        tags_after = []
        while True:
            words_after, tags_after = eval('case' + str(i))(sp)
            right = True
            for item in tags_after:
                if item not in ["R5","X","R6","R20","R99","R31","R7","R21","R30","R25","R90","R22","R23","R24"]:
                    right = False
            if right:
                break
        words_after_str = [''.join(words_after)]
        tags_after_str = ['RX']
        appendList(cat_v6_over2_20805_shuffle, words_first[j], tags_first[j], words_after_str, tags_after_str)
        appendList2(cat_v6_over2_20805_shuffle_after, words_after, tags_after)


def appendList(cat_v6_over2_20805_shuffle, words_first, tags_first, words_after, tags_after):
    words_fins = words_first + words_after
    tags_fins = tags_first + tags_after
    cat_v6_over2_20805_shuffle.append(','.join(words_fins) + '|' + ','.join(tags_fins))


def appendList2(cat_v6_over2_20805_shuffle_after, words_after, tags_after):
    words_fins = words_after
    tags_fins = tags_after
    cat_v6_over2_20805_shuffle_after.append(','.join(words_fins) + '|' + ','.join(tags_fins))


def strengthMain(PATH):
    sp = savePkl(PATH)
    cat_v6_over2_20805_shuffle = []
    cat_v6_over2_20805_shuffle_after = []
    y = 0
    y1 = 0
    fw = open(PATH + 'generate/add_sample_first_v210281751.txt', 'w', encoding='utf-8')
    fw2 = open(PATH + 'generate/add_sample_second_v210281751.txt', 'w', encoding='utf-8')
    splitAddressUtilR4(cat_v6_over2_20805_shuffle, sp, cat_v6_over2_20805_shuffle_after)
    random.shuffle(cat_v6_over2_20805_shuffle)
    random.shuffle(cat_v6_over2_20805_shuffle)
    random.shuffle(cat_v6_over2_20805_shuffle)
    random.shuffle(cat_v6_over2_20805_shuffle_after)
    random.shuffle(cat_v6_over2_20805_shuffle_after)
    random.shuffle(cat_v6_over2_20805_shuffle_after)
    for i in cat_v6_over2_20805_shuffle[:]:
        y += 1
        fw.write(i + '\n')
    for i in cat_v6_over2_20805_shuffle_after[:]:
        y1 += 1
        fw2.write(i + '\n')
    print('Generate data:', y, y1)
    fw.close()
    fw2.close()