import torch.utils.data as data
import torch
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms


class SCdataset(data.Dataset):
    def __init__(self, txt_path, transform=None):
        fh = open(txt_path, 'r')
        lists = []
        for line in fh:
            line = line.rstrip()
            words = line.split(' ')
            # lists.append((words[0], words[1], int(words[2])))
            lists.append((words[0], int(words[1])))

        self.lists = lists
        self.transform = transform

    def __getitem__(self, index):
        # imagename, maskname, label = self.lists[index]
        imagename, label = self.lists[index]

        img = Image.open(imagename) #.convert('RGB')
        # mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        if self.transform is not None:
            img = self.transform(img)
        # mask = cv2.resize(mask,(40,40))
        return img, label, imagename.split('/')[-1]

    def __len__(self):
        return len(self.lists)



######################## below is used for validation
def deTransform(mean, std, tensor):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor


if __name__ == '__main__':
    mean = [0.8106, 0.5949, 0.8088]
    std = [0.1635, 0.2139, 0.1225]
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])

    dataSet = SCdataset('/home/cyyan/projects/CaSoGP/data/train6.txt', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataSet, batch_size=1, shuffle=True, num_workers=4)
    for i, (input, label) in enumerate(train_loader):
        # print(i)
        # print(label)

        img = np.uint8(255*deTransform(mean, std, input).squeeze().numpy()) # 将颜色反标准化
        img = img[::-1,:,:].transpose((1,2,0)) # 转为opencv读取的格式 H*W*C， RGB --> BGR
        # mask = np.uint8(255*mask.squeeze().numpy())

        # cv2.imwrite('origin.png', img)
        # cv2.imwrite('mask.png', mask)
        # cv2.waitKey(0)
        break
        # print('wth')
