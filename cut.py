from PIL import Image
import PIL
import torchvision.transforms as transform

def cut_number(img_path,index,label):
    img = Image.open(img_path).convert('RGB')
    h=img.size[1]
    w=img.size[0]
    w_delta=w/5
    w_index=w_delta*index
    img=img.crop((w_index,0,w_index+w_delta,h))
    img=img.resize((28,28))
    r, g, b = img.split()
    img = r

    # test the dataloader
    # save_path="./test/"+img_path[32:(len(img_path)-4)]+str(index)+str(label)+'.png'
    # img.save(save_path)

    tran = transform.ToTensor()
    img = tran(img)
    return img

