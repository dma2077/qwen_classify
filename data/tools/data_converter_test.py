import json
import os
import argparse


def convert_food2k_jsonl(input_file, output_file, dataset_name="food101"):
    """
    将food2k_question.jsonl格式转换为目标JSONL格式
    输入格式: {"image": "path", "category": "label"}
    输出格式: {"image_path": "path", "label": "label", "dataset_name": "food101"}
    """
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return False
    
    try:
        converted_count = 0
        skipped_count = 0
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 逐行读取JSONL文件并转换
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                try:
                    # 解析JSON行
                    data = json.loads(line)
                    
                    # 检查必需的字段
                    if 'image' not in data:
                        print(f"第{line_num}行：跳过，缺少'image'字段")
                        skipped_count += 1
                        continue
                    
                    if 'category' not in data:
                        print(f"第{line_num}行：跳过，缺少'category'字段")
                        skipped_count += 1
                        continue
                    
                    # 提取数据
                    image_path = data['image']
                    label = data['category']
                    
                    # 检查数据有效性
                    if not image_path or not label:
                        print(f"第{line_num}行：跳过，image或category为空")
                        skipped_count += 1
                        continue
                    
                    # 构建输出格式
                    output_data = {
                        "image_path": image_path,
                        "label": label,
                        "dataset_name": dataset_name
                    }
                    
                    # 写入输出文件
                    outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    converted_count += 1
                    
                    # 每1000条打印一次进度
                    if converted_count % 1000 == 0:
                        print(f"已处理 {converted_count} 条数据...")
                
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行：JSON解析错误: {e}")
                    skipped_count += 1
                    continue
                except Exception as e:
                    print(f"第{line_num}行：处理错误: {e}")
                    skipped_count += 1
                    continue
        
        print(f"\n转换完成！")
        print(f"成功转换: {converted_count} 条数据")
        print(f"跳过: {skipped_count} 条数据")
        print(f"输出文件: {output_file}")
        return True
        
    except Exception as e:
        print(f"文件处理错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将food2k_question.jsonl转换为指定格式的JSONL')
    parser.add_argument('-i', '--input_file', help='输入JSONL文件路径')
    parser.add_argument('-o', '--output', help='输出JSONL文件路径（默认为输入文件名_converted.jsonl）')
    parser.add_argument('-d', '--dataset', default='food101', help='数据集名称（默认：food101）')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output
    
    # 如果没有指定输出文件，使用默认名称
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_test.jsonl"
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"数据集名称: {args.dataset}")
    print("-" * 50)
    
    success = convert_food2k_jsonl(input_file, output_file, args.dataset)
    if success:
        print("\n转换成功完成！")
    else:
        print("\n转换失败！")


if __name__ == "__main__":
    main()