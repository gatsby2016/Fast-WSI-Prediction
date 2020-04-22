import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import openslide
from scipy import io

import torch
import torch.nn.functional as F
from torchvision import transforms

import myModelVgg


def get_wsi_info(slidepath):
    slide = openslide.open_slide(slidepath)
    print('Numbers of level in this WSI: ', slide.level_count)
    print('Dimensions of all levels in this WSI (width, height):\n ', slide.level_dimensions)
    return slide.level_count, slide


def extract_wsi_tissue(slide, filter_level=9):
    """
    Args:
        slide: deepslide-format obtained by openslide.open_slide() method
        filter_level (int): the magnification for filtering the background of WSI image, Pyramid mapping method.
    Returns:
        low-level WSI image and binary mask with tissue and background separation by OTSU threshing method.
        binary mask only contains pixel value 0 (tissue region) and 1 (non-tissue, background)
    """
    width, height = slide.level_dimensions[filter_level]
    low_wsi = slide.read_region((0, 0), level=filter_level, size=(width, height))
    low_wsi = np.transpose(np.array(low_wsi)[:, :, 0:3], [1, 0, 2]) # get rgb from rgba and transpose
    low_wsi[low_wsi == 0] = 255 # fill the zero region to 255
    gray = cv2.cvtColor(low_wsi[..., [2, 1, 0]], cv2.COLOR_BGR2GRAY) # rgb2bgr then gray
    value, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU)
    return low_wsi, thresh


def fast_wsi_pred(slide, model, level, psize, blocksize, factor, numclass=2):
    """ fast WSI prediction by fully convolutional classification network.
    Args:
        slide: return deepslide-format obtained by openslide.open_slide() method
        model: parameters-loaded model architecture
        level (int): the magnification of trained patch. 0,1,2,3... denotes 40x, 20x, 10x, 5x ...
        psize (int): patch size of training sample, here we trained on patch shape (50, 50)
        blocksize (int): block size of testing ROI by fast-WSI-prediction method, depends on GPU util.
        factor (int): down-sampling factor of model you trained,
                If you have 3 max-pooling with 2*2 kernel size in whole model, factor is pow(2,3) = 8
        numclass (int): num of class for your classification task.
    Returns:
        variable `vis`, which contains WSI predicted probability value. Shape: (numclass, outsize_block, outsize_block)
    """
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977])])

    width, height = slide.level_dimensions[level]

    step = blocksize - psize + factor  # window sliding stride for testing block, which depends on 3 args.
    outsize_block = (blocksize - psize)//factor + 1  # output size of testing block flowing through the trained model

    vis = np.zeros((numclass, (width//step+1)*outsize_block, (height//step+1)*outsize_block)) # pre defined shape of vis
    print('Initial WSI prediction shape: ', vis.shape)

    for j in tqdm(range(0, height, step)):  # loop for the whole slide image from row to column.
        for i in range(0, width, step):
            image = slide.read_region((i * pow(2, level), j * pow(2, level)), level=level,
                                      size=(min(blocksize, width - i), min(blocksize, height - j)))
            # obtain each ROI block for testing, return the PIL format image.
            # remember that! the first arg (x, y) tuple of slide.read_region must in the level 0 reference frame.
            image = np.array(image)[:, :, 0:3]
            image = np.transpose(image, [1, 0, 2])
            # convert dimension to [1,0,2] due to dimension change between PIL.Image and numpy

            torch_img = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                torch_img = torch_img.cuda()
                pred_tensor = F.softmax(model(torch_img), dim=1)
                prob = pred_tensor.squeeze(0).cpu().detach().numpy()
            _, thisw, thish = prob.shape

            blocki = i//step
            blockj = j//step
            vis[:, blocki*outsize_block:blocki*outsize_block+thisw, blockj*outsize_block:blockj*outsize_block+thish] = prob
            # fill the predicted probabilty into pre-defined variable `vis`

    vis = vis[:, 0: vis.shape[1]-outsize_block+thisw, 0: vis.shape[2]-outsize_block+thish]
    # need to remove the predicted edge region without image content, understand with code in line 54.
    print('Final WSI prediction shape: ', vis.shape)
    return vis


def main(slide_path, model_path, level, psize, bsize, factor, n_lass, save_name,
         channel=1, filterFLAG=True, showFLAG=True):
    """ main function for fast WSI prediction
    Args:
        slide_path (str):
        model_path (str):
        level (int): the magnification of trained patch. 0,1,2,3... denotes 40x, 20x, 10x, 5x ...
        psize (int): patch size of training sample, here we trained on patch shape (50, 50, 3)
        bsize (int): block size of testing ROI by fast-WSI-prediction method, depends on GPU util.
        factor (int): down-sampling factor of model you trained,
                If you have 3 max-pooling with 2*2 kernel size in whole model, factor is 8
        n_lass (int): num of class for your classification task.
        save_name (str): name of file for saving variable `prob`, which is the WSI prediction probability.
        channel (int): show the channel belong to one specifc class for the classification task.
        filterFLAG (bool): whether filter the background of the WSI image or not `True`, filter it.
        showFLAG (bool): whether show the WSI predicted probability or not, `True`, show it; otherwise save into `save_name`
    Returns:
        show the WSI predicted probability map if showFLAG is `True';
        or, save the WSI predicted probability variable `prob` to save_name.
    """

    level_count, slide = get_wsi_info(slide_path)
    assert 0 <= level < level_count, 'magnification should be in the range of [0, {0}]'.format(level_count-1)

    net = myModelVgg.myVGG(in_channels=3, num_classes=2)
    net = net.cuda()     # net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    visual = fast_wsi_pred(slide, net, level=level, psize=psize, blocksize=bsize, factor=factor, numclass=n_lass)

    wsi_img, binary_tissue = extract_wsi_tissue(slide)
    if filterFLAG:
        binary_tissue = cv2.resize(binary_tissue, visual.shape[::-1][:2], interpolation=cv2.INTER_NEAREST)

        background = binary_tissue/2.38  # get background to probability 0.42, a middle value for better WSI visualization.
        binary_tissue = ~(binary_tissue*255)//255  # set value 0 for non-tissue and value 1 for tissue region
        visual = visual * binary_tissue + background

    visual = np.transpose(visual, [1, 2, 0])

    if showFLAG:
        plt.imshow(wsi_img, cmap='jet')
        plt.show()
        plt.imshow(visual[:, :, channel], cmap='jet')
        plt.show()
    else:
        np.save(save_name, visual)
        io.savemat(save_name.replace('.npy', '.mat'), {'prob': visual})
        plt.imsave(save_name.replace('.npy', '.png'), visual[:, :, channel], cmap='jet')
        # visual = np.load('../results/fullyConv/vis_FullyConv.npy')


if __name__ == "__main__":
    slidepth = '../data/4_201647135_3.ndpi'
    modelpth = '../models/FastWSI_vgg_epoch_95.pkl'
    magnification = 4  # the magnification of trained patch. 0,1,2,3... denotes 40x, 20x, 10x, 5x ...
    patch_size = 50  # patch size of training sample, here we trained on patch shape: 50*50*3
    block_size = 2048+2  # block size of testing ROI by fast-WSI-prediction method.
    factor = 8  # down-sampling factor of model you trained, Here we have 3 max-pooling with 2*2 kernel size in model.

    savename = '../results/FastWSI_Pred.npy'
    # main(slidepth, modelpth, magnification, patch_size, block_size, factor, 2, savename,
    #      channel=1, filterFLAG=False, showFLAG=False)

