import os
import re
import json
import tqdm
import math

import utils


def paras_xxx(json_path, shape_type="polygon"):
    with open(json_path, encoding="utf-8") as f:
        json_message = json.load(f)
    file_appear_class = []
    for shape in json_message["shapes"]:
        if shape["shape_type"] != shape_type:
            continue
        file_appear_class.append(shape["label"])
    for appear_class in sorted(set(file_appear_class)):
        yield appear_class


def make_sample_message(json_dir, connector="_"):
    # 归类
    json_dict = {}
    for root, dirs, names in os.walk(json_dir):
        for name in tqdm.tqdm(names, desc=os.path.split(root)[-1]):
            if re.search("^[a-zA-Z]*-[a-zA-Z]*-[0-9]*_[0-9]*\.json$", name) is None:
                continue
            prefix = name.split(connector)[0]
            if prefix not in json_dict.keys():
                json_dict[prefix] = []
            json_dict[prefix].append(os.path.join(root, name))

    # 统计
    outputs = []
    for prefix in tqdm.tqdm(json_dict.keys(), desc="statistic"):
        subset_message = {}

        subset_path = json_dict[prefix]
        subset_message["subset"] = prefix
        subset_message["images_count"] = len(subset_path)
        subset_message["tag"] = {}
        subset_message["tag"]["hospital"] = prefix.split("-")[1]
        subset_message["classes"] = {}

        for json_path in json_dict[prefix]:
            for label_name in paras_xxx(json_path):
                if "shafts" in label_name:
                    continue
                if "ignore" in label_name:
                    continue
                if "excess" in label_name:
                    continue
                if "background" in label_name:
                    continue
                if label_name == "":
                    continue
                label_name = label_name.split("_")[0].replace(" tip", "")
                if label_name not in subset_message["classes"].keys():
                    subset_message["classes"][label_name] = 0
                subset_message["classes"][label_name] += 1

        outputs.append(subset_message)
    return outputs


def std(*values):
    mean_value = sum(values) / len(values)
    return sum([(x - mean_value) ** 2 for x in values]) / len(values)


if __name__ == '__main__':
    samples_dir = "/mnt/FileExchange/withai/dataset/processed-data/lc-instruments-segmentation-0719/20-categories"
    tags_name = ["hospital"]
    dataset_number = 3
    datasets_ratio = [1 / dataset_number] * dataset_number
    class_important_sorting = ['claw grasper', 'grasping forcepss', 'specimen bag', 'aspirator', 'coagulator',
                               'straight dissecting forceps', 'atraumatic forceps', 'atraumatic fixation forceps',
                               'scissor', 'maryland dissecting forceps', 'metal clip', 'cautery hook', 'clip applier',
                               'gauze', 'absorbable clip']

    sample_set_message = make_sample_message(samples_dir)
    for x in sample_set_message:
        if "grasping forcepss" in x["classes"]:
            print(f"{x['subset']} -> {x['classes']['grasping forcepss']}")
    tag_divide_res = utils.divide_tag(sample_set_message, tags_name)
    tag_divide_res = {"hp": sample_set_message}
    classes_count = {}
    for sample_message in sample_set_message:
        for class_name in sample_message["classes"].keys():
            if class_name not in classes_count.keys():
                classes_count[class_name] = 0
            classes_count[class_name] += sample_message["classes"][class_name]

    min_divide_std = 1e16
    best_divide = {}
    for i in range(9999):
        divide_res = utils.divide_classes(classes_count, tag_divide_res, datasets_ratio, class_important_sorting)
        std_value = 0
        std_value += std(*[len(x["subsets"]) for x in divide_res["hp"].values()])

        for k in divide_res["hp"][0]["classes"]:
            std_value += std(*[x["classes"][k] for x in divide_res["hp"].values()])
        if std_value < min_divide_std:
            min_divide_std = std_value
            best_divide = divide_res
            print("{:0>5}-min_divide_std: {}".format(i, min_divide_std))

    print(best_divide)
    with open("best1.json", "w", encoding="utf-8") as f:
        json.dump(best_divide, f, ensure_ascii=False, indent=2)
    pass
