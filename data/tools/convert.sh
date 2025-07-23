#!/bin/bash

# 批量转换脚本：将category转换为label_id
# 输入目录：/llm_reco/dehua/code/qwen_classify/data/dataset_food
# 输出目录：/llm_reco/dehua/code/qwen_classify/data/dataset_food_label

# 设置输入和输出目录
INPUT_DIR="/llm_reco/dehua/code/qwen_classify/data/dataset_food"
OUTPUT_DIR="/llm_reco/dehua/code/qwen_classify/data/dataset_food_label"

# 设置转换脚本的路径（相对于当前工作目录）
CONVERTER_SCRIPT="convert_label.py"

# 检查转换脚本是否存在
if [ ! -f "$CONVERTER_SCRIPT" ]; then
    echo "错误: 转换脚本 $CONVERTER_SCRIPT 不存在！"
    echo "请确保在包含 convert_category_to_labelid.py 的目录中运行此脚本。"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 $INPUT_DIR 不存在！"
    exit 1
fi

# 创建输出目录（如果不存在）
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "创建输出目录: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
else
    echo "输出目录已存在: $OUTPUT_DIR"
fi

# 初始化计数器
total_files=0
successful_files=0
failed_files=0

echo "=========================================="
echo "开始批量转换文件"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "转换脚本: $CONVERTER_SCRIPT"
echo "=========================================="

# 查找所有jsonl文件并处理
for input_file in "$INPUT_DIR"/*.jsonl; do
    # 检查文件是否存在（处理没有匹配文件的情况）
    if [ ! -f "$input_file" ]; then
        echo "警告: 没有找到 .jsonl 文件在 $INPUT_DIR 目录中"
        break
    fi
    
    # 获取文件名（不包含路径）
    filename=$(basename "$input_file")
    
    # 构建输出文件路径
    output_file="$OUTPUT_DIR/$filename"
    
    echo ""
    echo "处理文件: $filename"
    echo "输入: $input_file"
    echo "输出: $output_file"
    
    # 执行转换
    if python3 "$CONVERTER_SCRIPT" "$input_file" "$output_file"; then
        echo "✓ 成功转换: $filename"
        ((successful_files++))
    else
        echo "✗ 转换失败: $filename"
        ((failed_files++))
    fi
    
    ((total_files++))
done

echo ""
echo "=========================================="
echo "批量转换完成！"
echo "总文件数: $total_files"
echo "成功转换: $successful_files"
echo "转换失败: $failed_files"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 如果有转换失败的文件，返回非零退出码
if [ $failed_files -gt 0 ]; then
    exit 1
else
    exit 0
fi