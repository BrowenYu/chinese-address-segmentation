import json

def json2dict(jsonStr, addr=None):
    if addr:
        with open(addr,'r') as load_f:
            load_dict = json.load(load_f)
    else:
        load_dict = json.loads(jsonStr)
    return load_dict

def dict2json(dictObj, addr=None):
    jsonStr = json.dumps(dictObj)
    if addr:
        g = open(addr, 'w')
        g.write(jsonStr)
        g.close()
        return
    else:
        return jsonStr

def txtReadJson(fileAddr):
    dict_csv = {}
    with open(fileAddr, 'r') as f:
        for line in f:
            line = line.strip()
            key, value = line.split('  ')
            dict_csv[key] = eval(value)
    return dict_csv

if __name__ == '__main__':
    # a = {
    #     '/cat_v6_over2.csv/part-00000-c4f38d0b-fc40-4efd-82ad-8726649e41a2-c000.csv':{'1':[3,4,5], '2':[10,11,12,13], '3':[17,18,19,20], '4':[24,25,26], '5':[-1,32]},
    #     '/cat_v6_over3.csv/part-00000-885a8b94-da52-4dff-b9eb-23ae1fd8201f-c000.csv':{'1':[3,4,5], '2':[10,11,12,13], '3':[17,18,19,20], '5':[21,22]},
    #     '/cat_v6_over4.csv/part-00000-2654456c-3ba3-4ad5-8df4-33cebf9b62f3-c000.csv':{ '1':[3,4,5], '2':[10,11,12,13], '4':[17,18,19], '5':[24,25,26]},
    #     '/cat_v6_over5.csv/part-00000-5bf141ad-45a8-409a-a191-c8759184f894-c000.csv':{'1':[3,4,5], '2':[10,11,12,13], '3':[17,18,19,20], '4':[24,25,26], '5':[28,29]},
    #     '/cat_v6_over6.csv/part-00000-aa37b8d5-95b1-435b-9397-ef90ae03ea98-c000.csv':{'1':[3,4,5], '3':[10,11,12,13], '4':[17,18,19], '5':[24,25]},
    #     '/cat_v6_over7.csv/part-00000-3255f4ba-5625-4df3-a5cc-29ae50a21094-c000.csv':{'1':[3,4,5], '3':[10,11,12,13], '5':[14,15]},
    #     '/cat_v6_over8.csv/part-00000-3aefe87d-9cad-4006-a03b-27d12b8c7f8d-c000.csv':{'1':[3,4,5], '3':[10,11,12,13], '4':[17,18,19], '5':[21,22]},
    #     '/cat_v6_over9.csv/part-00000-3611e20d-aadd-490f-8ae8-eb03d1b2d8f9-c000.csv':{'1':[3,4,5], '2':[10,11,12,13], '5':[14,15]},
    #     '/cat_v6_over10.csv/part-00000-dc8d90d0-81b0-4cf0-bd98-c65712b6f178-c000.csv':{'1':[3,4,5], '2':[10,11,12,13], '4':[17,18,19], '5':[21,22]}
    # }
    # dict2json(a, 'E:\\bangsun\\code\\data\\test.json')
    # p = json2dict('', 'E:\\bangsun\\code\\data\\test.json')
    # print(p)
    txtReadJson('E:/bangsun/code/data/csvList.txt')
    