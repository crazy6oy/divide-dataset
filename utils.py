import json
import os
import re
import random
import tqdm
import copy


# #############################################
# 样本全集信息格式：
# [{
#     "subset": "xxx",
#     "images_count": 1000,
#     "tag": {
# "hospital":"HX",
#         ...
#     }
#     "classes":{
#         "a":100,
#         "b":1000
#     }
# }]
#
# 数据划分核心流程：
#   1. 根据tag划分集合；
#   2. 在子子集中根据classes各类别数量划分多个数据集；
#   3. 汇总；
#   4. 人工微调。
# #############################################


def list_find_key_value(input, k, v):
    """
    在[{},{},{}, ...]中找有到所有有k且input[k]==v的{}.
    :param input: [{},{},{}, ...]
    :param k: key
    :param v: value
    :return: [{}, ...]
    """
    res = [x for x in input if (k in x and input[k] == v)]
    if res == []:
        return None
    else:
        return res


def statistic_semantic_seg_num(json_dir, connector="_", shape_type="polygon"):
    print("semantic statistic start!")
    output = {}

    for root, dirs, names in os.walk(json_dir):
        for name in tqdm.tqdm(names, desc=os.path.split(root)[-1]):
            if re.search("^[a-zA-Z]*-[a-zA-Z]*-[0-9]*_[0-9]*\.json$", name) is None:
                continue

            json_path = os.path.join(root, name)
            with open(json_path, encoding="utf-8") as f:
                json_message = json.load(f)
            file_appear_class = []
            for shape in json_message["shapes"]:
                if shape["shape_type"] != shape_type:
                    continue
                file_appear_class.append(shape["label"])
            for appear_class in sorted(set(file_appear_class)):
                if appear_class not in output.keys():
                    output[appear_class] = 0
                output[appear_class] += 1

    print("semantic statistic finish!")
    return output


def statistic_region_num(json_dir, shape_type="polygon"):
    outputs = {}
    for root, dirs, names in os.walk(json_dir):
        for name in tqdm.tqdm(names, desc=os.path.split(root)[-1]):
            if re.search("^[a-zA-Z]*-[a-zA-Z]*-[0-9]*_[0-9]*\.json$", name) is None:
                continue
            json_path = os.path.join(root, name)

            with open(json_path, encoding="utf-8") as f:
                json_msg = json.load(f)

            for msg in json_msg["shapes"]:
                if msg["shape_type"] != shape_type:
                    continue

                if msg["label"] not in outputs.keys():
                    outputs[msg["label"]] = 0
                outputs[msg["label"]] += 1

    print(outputs)


def divide_tag(sample_set_message, tags_name):
    outputs = {}
    for sample in sample_set_message:
        tag_str = "-".join([sample["tag"][x] for x in tags_name])
        if tag_str not in outputs.keys():
            outputs[tag_str] = []
        outputs[tag_str].append(sample)
    return outputs


def divide_classes(classes_count, tag_divide_res, datasets_ratio, class_important_sorting):
    """
    划分数据集
    根据样本全集信息、tag的划分结果、datasets的数量和比率以及各类别的重要性排序来划分数据集。
    :param classes_count: 各类别目标数
    :param tag_divide_res: 根据tag划分好的结果
    :param datasets_ratio: 各数据子集比率
    :param class_important_sorting: 样本子集各类别重要性排序，从重要到不重要
    :return: 各tags下数据集划分结果
    """
    tag_divide_res = copy.deepcopy(tag_divide_res)
    outputs = {}
    for tag in tag_divide_res.keys():
        outputs[tag] = {}
        for i, dataset_ratio in enumerate(datasets_ratio):
            outputs[tag][i] = {
                "subsets": [],
                "classes": {}
            }
            for class_name in class_important_sorting:
                outputs[tag][i]["classes"][class_name] = 0

        subsets = tag_divide_res[tag]
        for class_name in class_important_sorting:
            subsets_class = [x for x in subsets if class_name in x["classes"]]
            random.shuffle(subsets_class)
            for subset in subsets_class:
                for dataset_ind in outputs[tag].keys():
                    if outputs[tag][dataset_ind]["classes"][class_name] / classes_count[class_name] > \
                            datasets_ratio[
                                dataset_ind]:
                        continue
                    outputs[tag][dataset_ind]["subsets"].append(subset["subset"])
                    for subset_class_name in subset["classes"]:
                        outputs[tag][dataset_ind]["classes"][subset_class_name] += subset["classes"][subset_class_name]
                    # 删除
                    subsets.pop(subsets.index(subset))
                    break

    return outputs


if __name__ == '__main__':
    samples_dir = "/mnt/FileExchange/withai/dataset/processed-data/lc-instruments-segmentation-0719/20-categories"
    print(statistic_semantic_seg_num(samples_dir))
