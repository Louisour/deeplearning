from torch.cuda import Dataset
from PIL import Image
import os
class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir#函数内的全局变量
        self.label_dir = label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)#获取self.path上的image列表

    def __getitem__(self, index):#数据的编号
        img_name=self.img_path[index]#path里的元素名称
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)#相对路径
        img = Image.open(img_item_path)
        label=self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)
root_dir='afhq/train'
label_dir='cat'

cats_dataset=MyDataset(root_dir='./cats',label_dir='./cats_label')