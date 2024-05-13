import json
import os


def update_and_reorder_shapes(shapes):
    # 更新标签
    for shape in shapes:
        if shape['label'] == 'calf_kneeBack_l':
            shape['label'] = 'calf_l'
        elif shape['label'] == 'calf_kneeBack_r':
            shape['label'] = 'calf_r'

    # 获取特定标签的所有索引
    def get_indexes(label):
        return [i for i, shape in enumerate(shapes) if shape['label'] == label]

    # 对每个人重新排序
    for person_start_index in range(0, len(shapes), 23):  # 假设每个人有23个关节点
        for calf_label, foot_label in [('calf_l', 'foot_l'), ('calf_r', 'foot_r')]:
            calf_indexes = get_indexes(calf_label)
            foot_indexes = get_indexes(foot_label)
            # 确保索引在当前处理的人的范围内
            calf_index = next(
                (index for index in calf_indexes if person_start_index <= index < person_start_index + 23), None)
            foot_index = next(
                (index for index in foot_indexes if person_start_index <= index < person_start_index + 23), None)
            if calf_index is not None and foot_index is not None and calf_index > foot_index:
                shapes.insert(foot_index, shapes.pop(calf_index))


# 替换为您的目录路径
directory_path = '/home/tww/Datasets/real/户外跑步数据集/标注文件/22186129'

# 列出目录中的所有json文件
json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

for json_file in json_files:
    file_path = os.path.join(directory_path, json_file)
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    update_and_reorder_shapes(json_data['shapes'])

    with open(file_path, 'w') as file:
        json.dump(json_data, file, indent=4)

print('批量处理完成。')


# 替换为您的目录路径
directory_path = '/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119'

# 列出目录中的所有json文件
json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

for json_file in json_files:
    file_path = os.path.join(directory_path, json_file)
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    update_and_reorder_shapes(json_data['shapes'])

    with open(file_path, 'w') as file:
        json.dump(json_data, file, indent=4)

print('批量处理完成。')