import os
import shutil
import re
def save_code_files(source_file, destination_folder):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(source_file, 'r', encoding="utf-8") as file:
        content = file.read()
    match = re.search(r'from net\.(\w+) import Net', content)
    if match:
        model_name = match.group(1)
        model_file_path = os.path.join('net', f'{model_name}.py')
    dest_train_file_path = os.path.join(destination_folder, os.path.basename(__file__))

    # 复制文件
    shutil.copyfile(source_file, dest_train_file_path)
    shutil.copyfile(model_file_path, os.path.join(destination_folder, f'{model_name}.py'))