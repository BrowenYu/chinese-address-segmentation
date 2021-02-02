import numpy as np
import pandas as pd
import random
import pickle
import time

from utils.utils import deleteRandom, shuffle_first, del_first
from processing.PKLmodel import Pkl
from processing.strengthProducter import SequenceProducter
from setting import STRENGTH_RULE_FILENAMES
from setting import STRENGTH_RULE_PATH
from setting import RECOGNIZE_RULE_SAVE_PATH
from setting import NO_RECOGNIZE_RULE_SAVE_PATH
from setting import FIRST_SECOND_SAVE_PATH
from setting import FIRST_SAVE_PATH
from setting import SECOND_SAVE_PATH


def save_all_sequence(all_type_sequences, all_type_save_path):
    for sequences, path in zip(all_type_sequences, all_type_save_path):
        with open(path, 'w', encoding='utf-8')as f:
            for seq in sequences:
                f.write(seq + '\n')


def save_all_rule(rules, save_path):
    for rule, path in zip(rules, save_path):
        rule_df = pd.DataFrame({"分级模型":rule[2], "一级增强模式": rule[0], "二级增强模式": rule[1]})
        rule_df.to_excel(path)


def strength(all_pattern, predict_result):
    all_first_second_sequence = []
    all_first_sequence = []
    all_second_sequence = []

    recognize_first_rule = []
    recognize_second_rule = []
    recognize_predict_result = []
    no_recognize_first_rule = []
    no_recognize_second_rule = []
    no_recognize_predict_result = []

    pkl = Pkl()
    sequence_pro = SequenceProducter(pkl)
    print("相关数据加载完毕，开始生成增强数据")
    for _, pattern in enumerate(all_pattern):
        hand_pattern = pattern[0]
        RX_patterns = pattern[1]
        for temp_pattern in RX_patterns:
            try:
                RX_pattern = temp_pattern[0]
                nums = temp_pattern[1]
                handle = temp_pattern[2]
                sequences, sequences_hand, sequences_RX = sequence_pro.generateSequence(hand_pattern, RX_pattern, nums, handle)
                all_first_second_sequence.extend(sequences)
                all_first_sequence.extend(sequences_hand)
                all_second_sequence.extend(sequences_RX)
                
                recognize_first_rule.append(hand_pattern)
                recognize_second_rule.append(temp_pattern)
                recognize_predict_result.append(predict_result[_])
            except Exception as e:
                # raise e
                no_recognize_first_rule.append(hand_pattern)
                no_recognize_second_rule.append(temp_pattern)
                no_recognize_predict_result.append(predict_result[_])
                continue

    random.shuffle(all_first_second_sequence)
    random.shuffle(all_first_sequence)
    random.shuffle(all_second_sequence)
    strength_datas =  [all_first_second_sequence, all_first_sequence, all_second_sequence]
    rules = [[recognize_first_rule, recognize_second_rule, recognize_predict_result], 
            [no_recognize_first_rule, no_recognize_second_rule, no_recognize_predict_result]]
    return strength_datas, rules


def strengthMain():
    grade_model_predict_result = []
    all_pattern = []
    no_recognize_first_rule = []
    no_recognize_second_rule = []
    no_recognize_predict_result = []
    for f in STRENGTH_RULE_FILENAMES:
        file_name = STRENGTH_RULE_PATH + f
        strength_pattern = pd.read_excel(file_name)
        for index, row in strength_pattern.iterrows():
            if "分级模型" in row:
                predict_result_temp = row["分级模型"]
            else:
                predict_result_temp = ""
            first_temp = row["一级增强模式"]
            second_temp = row["二级增强模式"]
            if pd.isna(first_temp) or pd.isna(second_temp):
                continue
            try:
                first_rule = eval(first_temp)
                second_rule = eval(second_temp)
                # if isinstance(second_rule[-1], bool):
                if isinstance(second_rule[-1], bool) or not isinstance(second_rule[-1][-1], bool):
                    second_rule = [second_rule]
            except Exception as e:
                # raise e
                no_recognize_first_rule.append(first_temp)
                no_recognize_second_rule.append(second_temp)
                no_recognize_predict_result.append(predict_result_temp)
                continue
            grade_model_predict_result.append(predict_result_temp)
            all_pattern.append([first_rule, second_rule])
    # 得到数据。合并不能识别的规则，保存数据
    strength_datas, rules = strength(all_pattern, grade_model_predict_result)
    rules[1][0].extend(no_recognize_first_rule)
    rules[1][1].extend(no_recognize_second_rule)
    rules[1][2].extend(no_recognize_predict_result)
    print("增强数据生成完毕，开始保存")
    save_all_sequence(strength_datas,
                      [FIRST_SECOND_SAVE_PATH, FIRST_SAVE_PATH, SECOND_SAVE_PATH])
    
    save_all_rule(rules, 
                 [RECOGNIZE_RULE_SAVE_PATH, NO_RECOGNIZE_RULE_SAVE_PATH])