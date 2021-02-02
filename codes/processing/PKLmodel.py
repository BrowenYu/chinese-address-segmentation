import os

from setting import PATH
from setting import PKL_PATH
from utils.utils import readPkl
import copy


class Pkl(object):
    def __init__(self):
        self.tag_pkl_map = {}
        PKL_FILE_NAME_LIST = [file_name for file_name in os.listdir(PKL_PATH) 
                      if file_name.startswith("r") and file_name.endswith(".pkl")]
        for pkl_file_name in PKL_FILE_NAME_LIST:
            temp_name = pkl_file_name.split("_")
            file_name = "_".join(temp_name[:2])
            file_name = file_name.lower()
            tag_name = temp_name[0]
            tag_name = tag_name.lower()
            pkl_data = readPkl(PKL_PATH + pkl_file_name)
            if file_name in self.tag_pkl_map:
                self.tag_pkl_map[file_name].extend(pkl_data)
            else:
                self.tag_pkl_map[file_name] = copy.deepcopy(pkl_data)

            if tag_name in self.tag_pkl_map:
                self.tag_pkl_map[tag_name].extend(pkl_data)
            else:
                self.tag_pkl_map[tag_name] = copy.deepcopy(pkl_data)