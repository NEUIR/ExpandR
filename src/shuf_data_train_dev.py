import json
import random
import argparse


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))  
    return data

def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')  
            


def split_train_dev(args):
    data = load_jsonl(args.input_file)

    random_seed = 42
    random.seed(random_seed)
    random.shuffle(data)
    dev_size = args.dev_num
    dev_data = data[:dev_size]
    train_data = data[dev_size:]

    save_jsonl(dev_data, args.dev_file)
    save_jsonl(train_data, args.train_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, 
                        default=None,
                        help="The path of raw data (708,740)."
                        )
    parser.add_argument("--train_file", type=str, 
                        default=None,
                        help="The path of train data (637,866)."
                        )
    parser.add_argument("--dev_file", type=str, 
                        default=None,
                        help="The path of dev data (70,874)."
                        )
    parser.add_argument("--dev_num", type=str, 
                        default=None,
                        help="The number of dev data."
                        )
    
    
    args = parser.parse_args()
    split_train_dev(args)

if __name__ == "__main__":
    main()