# 需要加强的规则excel文件
STRENGTH_RULE_FILENAMES = ["截至0113增强规则整合.xlsx"]
# STRENGTH_RULE_FILENAMES = ["模型问题v1112-shc.xlsx",
#                            "模型问题-牛诗雅v1117.xlsx",
#                            "模型问题-林宇静v1112.xlsx",
#                            "模型问题v1110.xlsx",
#                            "模型问题-余博文v1112.xlsx"]
# STRENGTH_RULE_FILENAMES = ["规则测试文件.xlsx"]
# 保存成功识别的规则的文件的文件名, 必须是excel文件
RECOGNIZE_RULE_SAVE_FILENAME = "reco_test.xlsx"
# 保存未识别的规则的文件的文件名, 必须是excel文件
NO_RECOGNIZE_RULE_SAVE_FILENAME = "no_reco_test.xlsx"
# 一级二级数据合起来没有RX标签的文件名
FIRST_SECOND_SAVE_FILENAME = "first_second_v0121.txt"
# 一级模型增强数据保存的文件名
FIRST_SAVE_FILENAME = "first_v0121.txt"
# 二级模型增强数据保存的文件名
SECOND_SAVE_FILENAME = "second_v0121.txt"



PATH = './data/'

PKL_PATH = PATH + 'pkl/'

STRENGTH_RULE_PATH = PATH + "rule_file/"

RECOGNIZE_RULE_SAVE_PATH = PATH + "generate/rule/" + RECOGNIZE_RULE_SAVE_FILENAME

NO_RECOGNIZE_RULE_SAVE_PATH = PATH + "generate/rule/" + NO_RECOGNIZE_RULE_SAVE_FILENAME

FIRST_SECOND_SAVE_PATH = PATH + "generate/strength/" + FIRST_SECOND_SAVE_FILENAME

FIRST_SAVE_PATH = PATH + "generate/strength/" + FIRST_SAVE_FILENAME

SECOND_SAVE_PATH = PATH + "generate/strength/" + SECOND_SAVE_FILENAME

# 当前版本训练数据文件夹名称
DATA_EDITION = "strength"
# 增强数据所在路径
ADD_SAMPLE_PATH = PATH + 'generate/strength/' + DATA_EDITION
# 增强数据文件名
STRENGTH_DATA_FILE_NAME = []
# 76万条模型数据名称
DL_DATA_FIRST = "dl_data_v4_first_20805.txt"
# 生成训练数据保存路径
TRAIN_DATA = PATH + "generate/train_data/"