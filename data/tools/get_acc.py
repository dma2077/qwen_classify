from nltk.metrics.distance import edit_distance
import json
import re
from nltk.metrics.distance import edit_distance
from tqdm import tqdm
import sys

def build_food101_id2category():
    """
    构建 food101 的 ID→类别 映射。
    class_list.txt 中每行格式 "index class_name"，索引从0开始。
    """
    id2cat = {}
    label_file="/llm_reco/dehua/data/food_data/food-101/meta/labels.txt"
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            category = line.strip()
            id2cat[idx] = category.lower()
    return id2cat

def build_food172_id2category():
    """
    构建 Food172 数据集的 ID→类别 映射。
    文件每行一个类别名称，对应的索引从 0 开始；
    真实类别索引存储在图片路径中，但需要减 1。
    """
    id2cat = {}
    label_file="/llm_reco/dehua/data/food_data/VireoFood172/SplitAndIngreLabel/FoodList.txt"
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            id2cat[idx] = line.strip().lower()
    return id2cat

def build_fru92_id2category():
    """
    构建 fru92 的 ID→类别 映射。
    class_list.txt 中每行格式 "index class_name"，索引从0开始。
    """
    id2cat = {}
    label_file="/llm_reco/dehua/data/food_data/fru92_lists/fru_subclasses.txt"
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            category = line.strip()
            id2cat[idx] = category.lower().replace("_", " ")
    return id2cat

def build_food2k_id2category():
    id2cat = {}
    label_file="/llm_reco/dehua/data/food_data/Food2k_complete_jpg/food2k_label2name_en.txt"
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('--')
            # 若行中包含索引和类别名，则取类别名；否则整体作为类别
            if len(parts) >= 2:
                category = parts[1].replace('_', ' ')
            else:
                category = parts[0].replace('_', ' ')
            id2cat[idx] = category.lower()
    return id2cat


def build_veg200_id2category():
    """
    构建 veg200 的 ID→类别 映射。
    class_list.txt 中每行格式 "index class_name"，索引从0开始。
    """
    id2cat = {}
    label_file="/llm_reco/dehua/data/food_data/veg200_lists/veg_subclasses.txt"
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            category = line.strip()
            id2cat[idx] = category.lower()
    return id2cat

def build_foodx251_id2category():
    """
    构建 FoodX-251 的 ID→类别 映射。
    class_list.txt 中每行格式 "index class_name"，索引从0开始。
    """
    id2cat = {}
    label_file="/llm_reco/dehua/data/food_data/FoodX-251/annot/class_list.txt"
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            # 若行中包含索引和类别名，则取类别名；否则整体作为类别
            if len(parts) >= 2:
                category = parts[1].replace('_', ' ')
            else:
                category = parts[0].replace('_', ' ')
            id2cat[idx] = category.lower()
    return id2cat
