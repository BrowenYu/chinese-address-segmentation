# 地址模糊匹配说明文档
## 模型说明
___
## 增强模块说明
### 增强模块运行方式
在./data/rule_file/路径下放入需要编写规则的excel文件,在excel文件中编写规则，**在以 “一级增强模式” 的列编写一级模型规则。在以 “二级增强模式” 的列编写二级模型规则，一定要有这两个列其他无所谓**。

修改setting.py中的相关配置文件

运行main.py  

### setting配置说明
STRENGTH_RULE_FILENAMES: 增强规则所在excel文件名，需要完整文件名，以xlsx结尾。

RECOGNIZE_RULE_SAVE_FILENAME: 成功识别的规则保存的文件名，必须是excel文件，以xlsx结尾

NO_RECOGNIZE_RULE_SAVE_FILENAME: 不能识别的规则保存的文件名，必须是excel文件，以xlsx结尾

FIRST_SECOND_SAVE_FILENAME: 一级二级数据序列保存的文件名，txt文件，以.txt结尾

FIRST_SAVE_FILENAME: 一级模型增强数据序列保存文件名，txt文件，以.txt结尾

SECOND_SAVE_FILENAME: 二级模型增强数据序列保存文件名，txt文件，以.txt结尾

修改上面配置为自己想要的文件名

编写一级模型增强数据生成规则[...] 一级模型规则只能是一个list里面就是词语的规则，不能包含多个list，举例：["r1", "r3", "r4"]，不可以包含多条序列规则例如[["r1", "r3", "r4"], ["r3", "r1", "r2"]]不合规则。  

编写二级模型增强数据生成规则[[...第一条pkl序列规则], [...第二条pkl序列规则], [...], ...]，二级模型可以包含多条规则，比如  
```
                [
                    [
                        [
                            "r6", "r20", "r7"
                        ], 600, False
                    ],

                    [
                        [
                            "r0_location", {"r6": "楼群"}, "r25"
                        ], 600, False
                    ],

                    [
                        [
                            ("r1", {"R30": "mm"}), "r20"
                        ], 600, True
                    ],
                ]
```
二级模型也可以只有一条序列规则比如
```
                [
                    [
                        "r6", "r20", "r7"
                    ], 600, False
                ]
```
["r6", "r20", "r7"]代表生成序列的词语规则，600代表利用这一条规则生成总计600条数据，False代表与这一二级模型规则组合的一级模型序列是否进行重复，随即删除，乱序处理，False代表不进行处理。  

二级模型规则之间不会互相影响上述例子中有三个规则，三个规则都会按照自己的规则生成600条数据，总计1800条数据。与这1800条二级数据组合的一级数据都是随机按照传入的一级规则随机生成的，也就是说这1800条数据的一级数据也全部是不同的。

按照上面的例子在一级数据文件中会生成1800条一级模型数据，在二级数据文件也会生成1800条二级模型数据，在一级二级数据文件中保存整合的1800条一级二级数据。

数据会生成在data文件夹中

### data文件夹结构说明
data/generate
程序生成的数据所在路径

1. data/generate/rule/ :规则识别情况excel所在路径
2. data/generate/strength/ :增强数据txt所咋路径

data/pkl
所有程序需要的相关pkl文件所在路径

data/rule_file
使用者编写的规则excel所在路径
### 增强数据生成规则说明
一条增强数据规则包含在一个list中
在list中每个元素代表生成序列中的一个pkl词语，
比如list中有5个元素，生成的序列就是5个pkl组成的序列
对于list中的一个pkl元素，有四种生成方式:
1. 指定pkl类型随机获取pkl词语。例如想要这个pkl词语是属于R1，只需表示成 "r1" 即可，程序会随机从所有r1 pkl中随机取一个。想要这个元素为r0_location_1101.pkl文件里的，只需指定"r0_location"，程序会随机从所有以r0_location为开头的pkl文件中随机取一个pkl词语。
2. 指定pkl词语与类型。例如想要这个pkl元素就是 "城东办事处"，R24 可以采用字典指定{"R24": "城东办事处"}，程序会认定这个位置就是R24的城东办事处。字典中的value也可以是一个list里面可以包含多个pkl词语{"R24": ["城东办事处", "aaa", "bbb"]}代表这个位置想拿取这三个R24词中的一个作为这个位置的词语，程序会随机从这三个pkl词语中任取一个。还可以多个key{"R24": ["城东办事处", "aaa", "bbb"], "R6": "楼群"}，代表这个位置想拿取这三个R24词或者R6楼群总共四个词语中的一个作为这个位置的词。**对于想要指定以什么开头或者什么结尾的词语，可以使用 "\*办事处" 或者 "农业\*" 代表，举例：{"r24_addvillage": "\*办事处"} 代表取一个r24_addvillage pkl类型的词语并且这个词语以办事处结尾，{"r35": "农业\*"} 代表取一个r35 类型词语并且这个词语以农业开头，此规则复杂举例{"R24": ["\*办事处", "农业\*"] 代表取一个r24的词语这个词语随机以办事处结尾，以农业开头。暂不支持包含某个字符串取词。不建议写使用开头结尾规则，生成词语时间较久，能用别的方法建议使用别的方法。**
3. 指定多个规则随机取一个规则生成词语。例如想要这个位置的pkl词语既可以是 R24"城东办事处" 也可以是R6"楼群"，则规则为[{"R24": "城东办事处", "R6": "楼群"}]。再例如这个位置既可以是 R24"城东办事处" 也可以是R6"楼群" 还可以是从R7中随机取一个pkl词语，规则为[{"R24": "城东办事处"}, {"R6": "楼群"}, "r7"]，程序会随机从这三个规则中任选一个作为此位置生成pkl词语的规则。
4. 组合规则生成词语。例如这个位置的词应该是一个r1与r24组合的一个词语，如杭州市城北办事处，则规则为({"R1": "杭州"}, {"R24", "城北办事处"})。例二词语应该是r1中随机一个词与城北办事处，则规则为("r1", {"R24", "城北办事处"})   

**注：一个词语的规则，str只能指定setting.py文件中PKL_PATH路径下文件的名字如：r0_location_1101.pkl一定要指定为r0_location，指定的str类型pkl不存在则会报出异常。list代表多个规则取一个规则生成词语，list中每个元素都要是合逻辑的规则。dict代表指定词语生成词语。tuple代表组合规则生成词语。一级二级序列规则这四条规则都适用，这四条规则是对于规则序列中一个词语而言的。**
### 复杂规则举例
```
[["r0_location", {"R24": "城东办事处"}, ("r1", {"R24", "城北办事处"})], {"r6": ["楼群", "mm"]}, "r25", {"R1": "杭州市"}]
```
此规则代表序列  
> 第一个pkl可以是从r0_location类型pkl中随机一个或者是R24的城东办事处或者是R1随机pkl与R24城北办事出的组合  
    第二个pkl可以是R6的楼群或者mm两个pkl中随机一个  
    第三个pkl是从R25中随机一个  
    第四个pkl是R1的杭州市

## 进阶增强规则
**请在理解基本规则的前提下阅读此部分规则**

**此部分规则目的在于处理如下问题：**
> 部分数据格式如：“夜丰村夜丰58号”，“邦联实业公司山东邦联实业”，此类数据标记结果应该为 “夜丰村,R5,夜丰,R5,58号,R6”，“邦联实业公司,R20,13号,R6,山东邦联实业,R20”。如果需要对此类数据进行数据增强需要创造这种前后几个词有一定关联的数据。

#### 进阶规则在规则编写上有所区别，进阶规则的使用不会影响基本规则，所以未使用进阶规则的时候，基本规则的编写格式和以前完全一致。进阶规则是对原来生成的词语序列进行重新组合，所以将进阶规则称为**序列重组规则**

首先举例 “邦联实业公司,R20,13号,R6,山东邦联实业,R20” 如果要创造这种格式的数据，规则如下：

    ```
    一级规则：[]
    二级规则：[[("r20_简称", {"r20": "公司"}), "r6", "r1"], [{"word_part1": "1--"}, [1, 2, {"r20": [3, "word_part1"]}]], True], 500, False]
    ```
![avatar][base64str]
在这条二级规则中可以看到比原来规则编写方式多出来了a部分。

1. 首先说明d部分True代表使用此进阶规则进行词语重组，要使用重组规则True必须存在。

2. b部分称为 “待选词字典” ，这部分的作用是，指明后续重组序列时会使用到的词语的部分。字典中"word_part1": 
"1--",代表需要取基础规则生成序列[("r20_简称", {"r20": "公司"}), "r6", "r1"]，序列词语中的第一个词，获取除去这个词语最后两个字的其余部分 "1--"中1代表第一个词语 --代表去除这个词语的最后两个字，对应到“邦联实业公司”，就是取 ”邦联实业“ 命名为word_part1 这个key可以随意命名但是必须是字符串。如果后续重组序列需要使用到多个词语的一部分也可以写多个key与value。“-”只能放在词标号的前后
value举例：“1-”：取原生成序列第一个词语 去除最后一个字，“-2--”：取原生成序列第二个词语 去除它的第一个字与最后两个字。
3. c部分是一个数组，这部分作用是，指明如何进行序列的重组。[1, 2, {"r20": [3, "word_part1"]}]，第一个元素1的意思是这个位置取原生成序列的第一个词语，第二个元素2的意思是这个位置取原生成序列的第二个词语，第三个元素{"r20": [3, "word_part1"]}的key代表此位置词语的pkl级别，value [3, "word_part1"]代表使用原序列第三个词语与前面生成的 “待选词字典” 中的word_part1 key代表的词语进行组合作为此位置的词语。

对于上面的“邦联实业公司,R20,13号,R6,山东邦联实业,R20”例子，也可以写成如下规则：

    ```
    一级规则：[]
    二级规则：[["r20_简称", {"r20": "公司"}, "r6", "r1"], [{}, [{"r20": [1, 2]}, 3, {"r20": [4, 1]}]], True], 500, False]
    ```
某条增强数据如需使用序列重组规则，建议将这类增强数据单独放到一个excel，方便仔细检查规则是否有误，也可以单独验证规则是否能够通过。

[base64str]:data:image/png;base64,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