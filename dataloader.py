import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from cut import cut_number

class Imgdataset(Dataset):
    def __init__(self, path):
        super(Imgdataset, self).__init__()
        print("dataloader")
        self.data = []
        if os.path.exists(path):
            dir_list = sorted(os.listdir(path))
            img_dir = 'easy_samples'  # !!!
            label_dir = 'easy_samples.txt'
            img_path = path + '/' + img_dir
            label_file = path + '/' + label_dir
            f=open(label_file,'r')
            labels_datas=f.read()
            f.close()
            if os.path.exists(img_path) and os.path.exists(label_file):
                img_data = sorted(os.listdir(img_path))
                for i in range(len(img_data)):
                    if img_data[i] =='.DS_Store':
                        continue
                    line = labels_datas.find("./easy_samples/"+img_data[i])
                    if line == -1:
                        continue
                    label_data=[]
                    for j in range(line+16+len(img_data[i]),line+31+len(img_data[i])):
                        if labels_datas[j]=='\n':
                            break
                        label_data.append(labels_datas[j])
                    for num in range(5):
                        self.data.append({'num': num,'img': img_path+'/'+img_data[i],'label': label_data})
            else:
                raise FileNotFoundError('path doesnt exist!')


    def __getitem__(self, index):
        num,img_path, label_datas = self.data[index]['num'],self.data[index]["img"], self.data[index]["label"]
        #label
        cnt=0
        label=[]
        for i in range(len(label_datas)):
            if label_datas[i]==',':
                cnt=cnt+1
                continue
            if cnt==num:
                label.append(label_datas[i])
            if cnt>num:
                break
        if len(label)==2:
            label_number=float(label[1])+0.5
        else:
            label_number=float(label[0])
        #img
        img=cut_number(img_path,num,label_number)

        return img, label

    def __len__(self):
        return len(self.data)