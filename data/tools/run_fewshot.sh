#!/bin/bash

# 数据集列表
datasets=("food101" "food172" "food2k" "fru92" "veg200" "foodx251")

# 输入数据目录
input_dir="/llm_reco/dehua/data/food_finetune_data"

# 输出数据目录
output_dir="/llm_reco/dehua/code/qwen_classify/data/dataset_food"

# 确保输出目录存在
mkdir -p "$output_dir"

echo "开始批量转换数据集..."
echo "输入目录: $input_dir"
echo "输出目录: $output_dir"
echo "=" * 60

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    echo "处理数据集: $dataset"
    
    # 处理4shot版本
    input_file_4shot="$input_dir/${dataset}_cold_sft_4shot.json"
    output_file_4shot="$output_dir/${dataset}_train_4shot.jsonl"
    
    if [ -f "$input_file_4shot" ]; then
        echo "  转换4shot版本..."
        python /llm_reco/dehua/code/qwen_classify/data/tools/data_converter.py \
            -i "$input_file_4shot" \
            -o "$output_file_4shot" \
            -d "$dataset"
        echo "  4shot转换完成: $output_file_4shot"
    else
        echo "  警告: 4shot输入文件不存在: $input_file_4shot"
    fi
    
    # 处理8shot版本
    input_file_8shot="$input_dir/${dataset}_cold_sft_8shot.json"
    output_file_8shot="$output_dir/${dataset}_train_8shot.jsonl"
    
    if [ -f "$input_file_8shot" ]; then
        echo "  转换8shot版本..."
        python /llm_reco/dehua/code/qwen_classify/data/tools/data_converter.py \
            -i "$input_file_8shot" \
            -o "$output_file_8shot" \
            -d "$dataset"
        echo "  8shot转换完成: $output_file_8shot"
    else
        echo "  警告: 8shot输入文件不存在: $input_file_8shot"
    fi
    
    echo "  数据集 $dataset 处理完成"
    echo "-" * 40
done

echo "所有数据集转换完成！"
echo "=" * 60