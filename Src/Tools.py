# -*- coding: utf-8 -*-
import os
import pandas as pd

#删除假代码X
def remove_x_from_file(filename):
    # 读取原始文件并删除所有 'X' 字符
    with open(filename, 'r') as file:
        content = file.read()
        content_without_x = content.replace('X', '')
    # 将处理后的数据写入临时文件
    with open('./Data/TempData/temp_cleaned.txt', 'w') as cleaned_file:
        cleaned_file.write(content_without_x)
#特征提取
def process_file(filename):
    # 复制原文件
    copy_txt_file(filename, "./Data/TempData/copy.txt")
    # 删除文件中的所有 'X' 字符
    remove_x_from_file(filename)
    # 运行Python脚本
    os.system('python ./Pse-in-One-2.0/nac.pyc ./Data/TempData/temp_cleaned.txt Protein DR -max_dis 1 -f tab -labels 0 -out ./Data/TempData/temp.txt')
    # 读取txt文件并转换为csv文件
    df = pd.read_csv('./Data/TempData/temp.txt', sep='\t')
    df.to_csv('./Data/TempData/featured.csv', index=False)
    print("特征提取成功！")
# -*- coding: utf-8 -*-
#copy一份原始数据集
def copy_txt_file(source_file, destination_file):
    try:
        with open(source_file, 'r') as source:
            with open(destination_file, 'w') as destination:
                for line in source:
                    destination.write(line)
        print("文件复制成功")
    except IOError:
        print("文件复制失败")


#输出偶数行内容
def print_even_lines(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            for i in range(1, len(lines), 2):  # 从索引 1 开始，步长为 2，即偶数行
                print(lines[i].strip())
    except IOError:
        print("文件未找到或无法读取")








