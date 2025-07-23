import json
import re
import os
import argparse


def extract_label_from_content(content):
    """从assistant的content中提取<answer>标签内的内容作为label"""
    answer_match = re.search(r'<answer>(.*?)</answer>', content)
    if answer_match:
        return answer_match.group(1).strip()
    return None


def convert_json_to_jsonl(input_file, output_file, dataset_name="food101"):
    """将JSON文件转换为JSONL格式"""
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return False
    
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果data是单个对象，将其放入列表中
        if isinstance(data, dict):
            data = [data]
        
        converted_count = 0
        skipped_count = 0
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 转换并写入JSONL文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data):
                try:
                    # 提取image_path
                    if 'images' not in item or not item['images']:
                        print(f"第{i+1}条：跳过，缺少images字段")
                        skipped_count += 1
                        continue
                    image_path = item['images'][0]  # 取第一个图片路径
                    
                    # 提取label
                    if 'messages' not in item or len(item['messages']) < 2:
                        print(f"第{i+1}条：跳过，缺少足够的messages")
                        skipped_count += 1
                        continue
                    
                    assistant_message = None
                    for msg in item['messages']:
                        if msg.get('role') == 'assistant':
                            assistant_message = msg
                            break
                    
                    if not assistant_message:
                        print(f"第{i+1}条：跳过，未找到assistant消息")
                        skipped_count += 1
                        continue
                    
                    label = extract_label_from_content(assistant_message['content'])
                    if not label:
                        print(f"第{i+1}条：跳过，无法提取label，content: {assistant_message['content'][:100]}...")
                        skipped_count += 1
                        continue
                    
                    # 构建JSONL格式的数据
                    jsonl_item = {
                        "image_path": image_path,
                        "label": label,
                        "dataset_name": dataset_name
                    }
                    
                    # 写入JSONL文件
                    f.write(json.dumps(jsonl_item, ensure_ascii=False) + '\n')
                    converted_count += 1
                    
                    # 每1000条打印一次进度
                    if converted_count % 1000 == 0:
                        print(f"已处理 {converted_count} 条数据...")
                    
                except Exception as e:
                    print(f"第{i+1}条：处理项目时出错: {e}")
                    skipped_count += 1
                    continue
        
        print(f"\n转换完成！")
        print(f"成功转换: {converted_count} 条数据")
        print(f"跳过: {skipped_count} 条数据")
        print(f"输出文件: {output_file}")
        return True
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将JSON文件转换为JSONL格式')
    parser.add_argument('-i' ,'--input_file', help='输入JSON文件路径')
    parser.add_argument('-o', '--output', help='输出JSONL文件路径（默认为输入文件名.jsonl）')
    parser.add_argument('-d', '--dataset', default='food101', help='数据集名称（默认：food101）')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output
    
    # 如果没有指定输出文件，使用默认名称
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.jsonl"
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"数据集名称: {args.dataset}")
    print("-" * 50)
    
    success = convert_json_to_jsonl(input_file, output_file, args.dataset)
    if success:
        print("\n转换成功完成！")
    else:
        print("\n转换失败！")


if __name__ == "__main__":
    main()