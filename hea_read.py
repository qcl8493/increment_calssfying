import os
dataset_dir = "E:/increment_classfying/dataset"

hea_files = [file for file in os.listdir(dataset_dir) if file.endswith('.hea')]
hea_files = sorted(hea_files, key=lambda x: int(x.split('S')[1].split('.')[0]))


count = 0
for hea_file in hea_files:
    hea_file_path = os.path.join(dataset_dir, hea_file)
    with open(hea_file_path, 'r') as f:
        content = f.read()
        print(content)
        print("----------------------------------------")
        count += 1
    if count > 910:
        break