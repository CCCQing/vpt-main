import json, os, random
from collections import defaultdict

# === 改成你的 CUB 根目录（包含 images/ 与 *.txt 的那个文件夹）===
ROOT = r"D:\postgraduate1\project\datasets\CUB\CUB_200_2011"

random.seed(1)  # 让 val 切分可复现（从官方 train 中按每类取10%做 val）

# 读官方索引文件
def read_kv(fname, key_cast=int, val_cast=str):
    d = {}
    with open(os.path.join(ROOT, fname), "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split()
            d[key_cast(a)] = val_cast(b)
    return d

# image_id -> 相对 images/ 的路径（例如 '001.xxx/xxx.jpg'，用 '/'）
images = read_kv("images.txt", int, str)
# image_id -> 类别ID（1..200）
labels = read_kv("image_class_labels.txt", int, int)
# image_id -> 1(train) / 0(test)
splits = read_kv("train_test_split.txt", int, int)

def kv_record(img_id):
    # JSON 的 key：相对 CUB_200_2011 的路径（强制用正斜杠）
    key = images[img_id]
    key = key.replace("\\", "/")
    # JSON 的 value：0-based 类别 ID
    val = int(labels[img_id] - 1)
    return key, val

# 官方提供的 train/test
train_ids = [i for i in images if splits[i] == 1]
test_ids  = [i for i in images if splits[i] == 0]

# 从官方 train 再划出 val：每类 10%
by_cls = defaultdict(list)
for iid in train_ids:
    by_cls[labels[iid] - 1].append(iid)

val_ids, new_train_ids = [], []
for c, lst in by_cls.items():
    lst = sorted(lst)
    k = max(1, int(0.1 * len(lst)))
    random.shuffle(lst)
    val_ids += lst[:k]
    new_train_ids += lst[k:]

def dump_json_kv(iids, fname):
    D = {}
    for iid in sorted(iids):
        k, v = kv_record(iid)
        D[k] = v
    with open(os.path.join(ROOT, fname), "w", encoding="utf-8") as f:
        json.dump(D, f, ensure_ascii=False)
    print(fname, len(D))

dump_json_kv(new_train_ids, "train.json")
dump_json_kv(val_ids,        "val.json")
dump_json_kv(test_ids,       "test.json")
