import json
import os
from pathlib import Path
import pandas as pd

def count_train_pairs(training_dir: str) -> None:
    """
    统计 training 目录下每个 JSON 文件中 train 字段的样本对数量
    """
    training_path = Path(training_dir)
    
    if not training_path.exists():
        print(f"Directory {training_dir} does not exist!")
        return
    
    # 获取所有 .json 文件
    json_files = list(training_path.glob("*.json"))
    json_files.sort()  # 按文件名排序
    
    total_files = len(json_files)
    total_pairs = 0
    
    print(f"Found {total_files} JSON files in {training_dir}")
    print("=" * 60)

    all_data={}
    
    for json_file in json_files:
        task_name=json_file.stem
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 统计 train 字段中的样本对数量
            train_pairs = len(data.get('train', []))
            total_pairs += train_pairs
            all_data[task_name]=train_pairs
            
        except Exception as e:
            print("出错了")
    
    print("=" * 60)
    print(f"Summary: {total_files} files, {total_pairs} total training pairs")
    print(".2f")
    try:
        df=pd.DataFrame([all_data])
        df.to_csv("./countexample.csv",index=True)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    # 修改这里为你实际的 training 数据目录路径
    training_directory = "/root/localdisk/VARC/raw_data/ARC-AGI/data/training"
    count_train_pairs(training_directory)