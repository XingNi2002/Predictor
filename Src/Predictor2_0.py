# -*- coding: utf-8 -*-
import codecs
import csv
import pickle
import Tkinter as tk
import tkFileDialog
import tkMessageBox
import Tools

def select_file():
    file_path = tkFileDialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        print("选择的文件路径:", file_path)
        Tools.process_file(file_path)
        # 处理完文件后关闭窗口
        root.destroy()

# 创建主窗口
root = tk.Tk()
root.title("Protein Sequence Prediction")

# 设置窗口大小
window_width = 512
window_height = 512
root.geometry("{}x{}".format(window_width, window_height))

# 创建按钮
button = tk.Button(root, text="选择 .txt 文件", command=select_file, width=20, height=2)
button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# 进入主事件循环
root.mainloop()

# 加载模型
with open("new_ensemble.pkl", "rb") as model_file:
    ensemble_model = pickle.load(model_file)

# 读取测试集
def read_csv(file_path):
    features = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            features.append([float(x) for x in row])
    return features

# 加载特征数据集
test_features = read_csv("./Data/TempData/featured.csv")

# 进行预测
predictions = ensemble_model.predict(test_features)
probabilities = ensemble_model.predict_proba(test_features)

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.page_index = 0
        self.init_window()

    def init_window(self):
        # 窗口大小
        self.master.title("Protein Sequence Prediction")
        self.master.geometry("{}x{}".format(window_width, window_height))
        # 显示文本框大小
        self.text_box = tk.Text(self.master, height=30, width=60)
        self.text_box.pack()

        # 上一页按钮
        self.prev_page_button = tk.Button(self.master, text="Previous Page", command=self.prev_page, padx=0, pady=5)
        self.prev_page_button.pack(side=tk.LEFT, padx=25, pady=5)

        # 下一页按钮
        self.next_page_button = tk.Button(self.master, text="Next Page", command=self.next_page, padx=0, pady=5)
        self.next_page_button.pack(side=tk.LEFT, padx=25, pady=5)

        # 退出按钮
        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit_program, padx=20, pady=5)
        self.quit_button.pack(side=tk.RIGHT, padx=25, pady=5)

        # 保存按钮
        self.save_button = tk.Button(self.master, text="Save All", command=self.save_all_to_file, padx=20, pady=5)
        self.save_button.pack(side=tk.RIGHT, padx=25, pady=5)

        self.update_content()


    def update_content(self):
        self.text_box.delete(1.0, tk.END)
        start_index = self.page_index * 20
        end_index = min(start_index + 20, len(lines))
        for i in range(start_index, end_index):
            line_number = i + 1
            if line_number % 2 == 0:  # 偶数行
                self.text_box.insert(tk.END, lines[i])

    def save_all_to_file(self):
        file_path = tkFileDialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with codecs.open(file_path, "w", encoding="utf-8") as file:
                for page_index in range(len(lines) // 20 + 1):
                    self.page_index = page_index
                    self.update_content()
                    content = self.text_box.get("1.0", "end")  # 获取整个文本框的内容
                    file.write(content)
            tkMessageBox.showinfo("保存成功", "数据已保存到文件 {}".format(file_path))

    def next_page(self):
        if (self.page_index + 1) * 20 < len(lines):
            self.page_index += 1
            self.update_content()

    def prev_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self.update_content()

    def quit_program(self):
        self.master.destroy()


def print_even_lines(filename):
    try:
        with open(filename, 'r') as file:
            global lines
            lines = file.readlines()
            protein_count = 1
            for i in range(1, len(lines), 2):  # 从索引 1 开始，步长为 2，即偶数行
                line = lines[i].strip() + '\n'
                line = "Protein {}: ".format(protein_count) + line  # 在蛋白质序列前添加 "Protein X"
                protein_count += 1  # 增加 X 的值
                # 打印对应样本的概率信息
                line += print_probabilities(i)
                lines[i] = line  # 替换原始列表中的数据
            root = tk.Tk()
            app = Application(master=root)
            app.mainloop()

    except IOError:
        print("文件未找到或无法读取")


def print_probabilities(start_index):
    start_index = float(start_index) / 2  # 使用浮点数除法
    start_index = int(start_index)  # 将结果转换回整数
    if probabilities[start_index][1] >= 0.5:
        return "{:.3f} 正样本\n\n".format(probabilities[start_index][1])
    else:
        return "{:.3f} 负样本\n\n".format(probabilities[start_index][1])

# 调用函数打印偶数行内容和概率信息
print_even_lines("./Data/TempData/copy.txt")


# 输出统计结果
count_0 = sum(1 for prediction in predictions if prediction == 0)
count_1 = sum(1 for prediction in predictions if prediction == 1)
print("预测结果中负样本的个数: {}".format(count_0))
print("预测结果中正样本的个数: {}".format(count_1))
