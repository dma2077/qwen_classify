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
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('--')
            # 若行中包含索引和类别名，则取索引和类别名
            if len(parts) >= 2:
                try:
                    # 第一部分应该是索引
                    idx = int(parts[0])
                    category = parts[1].replace('_', ' ')
                except ValueError:
                    # 如果第一部分不是数字，则使用枚举索引
                    print(f"警告: food2k标签文件格式异常: {line}")
                    continue
            else:
                # 如果没有--分隔符，使用枚举索引
                print(f"警告: food2k标签文件格式异常: {line}")
                continue
            
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
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            # 若行中包含索引和类别名，则取索引和类别名
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    category = ' '.join(parts[1:]).replace('_', ' ')  # 处理可能有多个单词的情况
                except ValueError:
                    print(f"警告: FoodX-251标签文件格式异常: {line}")
                    continue
            else:
                print(f"警告: FoodX-251标签文件格式异常: {line}")
                continue
            
            id2cat[idx] = category.lower()
    return id2cat

def build_category2id_mapping(dataset_name):
    """
    根据数据集名称构建 category→ID 映射
    """
    dataset_builders = {
        'food101': build_food101_id2category,
        'food172': build_food172_id2category,
        'fru92': build_fru92_id2category,
        'food2k': build_food2k_id2category,
        'veg200': build_veg200_id2category,
        'foodx251': build_foodx251_id2category
    }
    
    if dataset_name not in dataset_builders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 获取id2category映射
    id2cat = dataset_builders[dataset_name]()
    
    # 反转映射得到category2id，并对所有类别名称进行标准化
    cat2id = {normalize_text(cat): idx for idx, cat in id2cat.items()}
    
    return cat2id, id2cat

def normalize_text(text):
    """
    标准化文本：转小写，将下划线转为空格，去除多余空格
    """
    # 转小写
    text = text.lower().strip()
    # 将下划线转为空格
    text = text.replace('_', ' ')
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    return text

def find_best_match(target_label, cat2id_mapping, threshold=2):
    """
    使用编辑距离找到最佳匹配的category
    """
    target_normalized = normalize_text(target_label)
    
    # 首先尝试精确匹配
    if target_normalized in cat2id_mapping:
        return cat2id_mapping[target_normalized]
    
    # 如果精确匹配失败，使用编辑距离找最相似的
    best_match = None
    min_distance = float('inf')
    
    for category in cat2id_mapping.keys():
        distance = edit_distance(target_normalized, category)
        if distance < min_distance:
            min_distance = distance
            best_match = category
    
    # 如果最小编辑距离在阈值内，返回匹配结果
    if min_distance <= threshold:
        if min_distance > 0:  # 只对非精确匹配显示信息
            print(f"模糊匹配: '{target_label}' (标准化: '{target_normalized}') -> '{best_match}' (编辑距离: {min_distance})")
        return cat2id_mapping[best_match]
    
    # 如果没有找到合适的匹配，显示一些可能的候选
    candidates = []
    for category in cat2id_mapping.keys():
        distance = edit_distance(target_normalized, category)
        if distance <= threshold + 2:  # 显示距离稍大的候选
            candidates.append((category, distance))
    
    candidates.sort(key=lambda x: x[1])  # 按距离排序
    candidates_str = ", ".join([f"'{cat}'({dist})" for cat, dist in candidates[:5]])
    
    print(f"警告: 无法匹配标签 '{target_label}' (标准化: '{target_normalized}')")
    if candidates:
        print(f"  可能的候选 (距离): {candidates_str}")
    
    return None

def convert_jsonl_category_to_labelid(input_file, output_file, edit_distance_threshold=2):
    """
    将jsonl文件中的category标签转换为label_id
    """
    successful_conversions = 0
    failed_conversions = 0
    dataset_mappings = {}
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(tqdm(infile, desc="转换进度"), 1):
            try:
                data = json.loads(line.strip())
                
                # 获取必要字段
                dataset_name = data.get('dataset_name', '').lower()
                label = data.get('label', '')
                
                if not dataset_name or not label:
                    print(f"第{line_num}行: 缺少必要字段 dataset_name 或 label")
                    failed_conversions += 1
                    continue
                
                # 为每个数据集构建一次映射（缓存）
                if dataset_name not in dataset_mappings:
                    try:
                        cat2id, id2cat = build_category2id_mapping(dataset_name)
                        dataset_mappings[dataset_name] = cat2id
                        print(f"为数据集 '{dataset_name}' 构建了 {len(cat2id)} 个类别的映射")
                    except ValueError as e:
                        print(f"第{line_num}行: {e}")
                        failed_conversions += 1
                        continue
                
                # 查找对应的label_id
                cat2id_mapping = dataset_mappings[dataset_name]
                label_id = find_best_match(label, cat2id_mapping, edit_distance_threshold)
                
                if label_id is not None:
                    # 更新数据
                    data['original_label'] = label  # 保留原始标签作为参考
                    data['label'] = label_id  # 替换label字段为label_id
                    
                    # 写入转换后的数据
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    successful_conversions += 1
                else:
                    failed_conversions += 1
                    
            except json.JSONDecodeError:
                print(f"第{line_num}行: JSON解析错误")
                failed_conversions += 1
            except Exception as e:
                print(f"第{line_num}行: 处理错误 - {e}")
                failed_conversions += 1
    
    print(f"\n转换完成!")
    print(f"成功转换: {successful_conversions} 条")
    print(f"转换失败: {failed_conversions} 条")
    print(f"输出文件: {output_file}")

def main():
    if len(sys.argv) != 3:
        print("用法: python convert_category_to_labelid.py <input_jsonl_file> <output_jsonl_file>")
        print("示例: python convert_category_to_labelid.py test_food2k.jsonl test_food2k_with_labelid.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 设置编辑距离阈值，经过标准化处理后可以使用更小的阈值
    edit_distance_threshold = 1
    
    convert_jsonl_category_to_labelid(input_file, output_file, edit_distance_threshold)

if __name__ == "__main__":
    main()