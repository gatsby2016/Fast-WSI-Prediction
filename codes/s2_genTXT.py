from glob import glob
import os
import random


def gen_txt(txt_path, img_dir, sublist):
    f = open(txt_path, 'w')
    for sp in sublist:
        for root, s_dirs, _ in os.walk(os.path.join(img_dir, sp), topdown=True):  # 获取 train文件下各文件夹名称
            for folder_ind in s_dirs:
                i_dir = os.path.join(root, folder_ind)  # 获取各类的文件夹 绝对路径
                img_list = os.listdir(i_dir)  # 获取类别文件夹下所有png图片的路径
                for i in range(len(img_list)):
                    if not img_list[i].endswith('png'):  # 若不是png文件，跳过
                        continue
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + ' ' + folder_ind + '\n'
                    f.write(line)
    f.close()
    print(txt_path, 'finished!')


if __name__ == "__main__":
    random.seed(3)
    path = '../data/IDCall/'
    subpath = os.listdir(path)
    random.shuffle(subpath)

    train_num = int(len(subpath)*0.6)
    val_num = int(len(subpath)*0.2)

    traintxt = '../data/train.txt'
    gen_txt(traintxt, path, subpath[0: train_num])
    valtxt = '../data/val.txt'
    gen_txt(valtxt, path, subpath[train_num: train_num+val_num])
    testtxt = '../data/test.txt'
    gen_txt(testtxt, path, subpath[train_num+val_num:])