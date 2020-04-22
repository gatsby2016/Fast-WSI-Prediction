# Fast WSI Prediction  
**Fast WSI prediction by ScanNet in PyTorch.**  


## Aim: undergraduate course  
**Invasive Ductal Carcinoma (IDC) detection** on Whole Slide Image (WSI) by deep learning method eqipped with **[ScanNet](https://ieeexplore.ieee.org/abstract/document/8354169/) fully conv. scheme** for fast WSI prediction.  

  
## Data  
**279 slides** with IDC ROI annotations.  
Ratio of **6:2:2** for training, validation and testing separation.  

Tiling small patch with 50\*50 at x2.5 magnification for `non-IDC` & `IDC` class, binary classification task.  

|** patches distribution** |  non-IDC  |  IDC  |
| -------------------------|  -----    | ----- |
| training                 | 111090    | 49204 |
| validation               | 45356     | 15354 |
| testing                  | 42292     | 14228 |


## Method
1. Replace the original `VGG` with `VGG_fullyConv` for Fast WSI prediction  
2. Train the `VGG_fullyConv` model for classification, the same as original training process  
3. Check the performance on `Validation` dataset and select the best in terms of F1 score  
4. Test and evaluate your trained `VGG_fullyConv` model on `Testing` dataset in patch-level  
5. Infer and predict the probability map for one WSI and show it!  

### HotSpot  
> We significantly improve the inferring time by this [ScanNet](https://ieeexplore.ieee.org/abstract/document/8354169) scheme.  

### Implementation
Replaced **the last GAP and fc.** in `VGG` with **AvgPooling with 2x2 kernel size followed by 2 convs  with 1x1 kernel**, [s6_predWSI.py](https://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/codes/s6_predWSI.py)    
```python
self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

self.classifier = nn.Sequential(
    nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
    nn.BatchNorm2d(out_channel),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channel, num_classes, kernel_size=1, stride=1))
```   

Also, we modify the `VGG` network due to our small training size. You can refer to [myModelVgg.py](https://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/codes/myModelVgg.py) for more details.  


**Noting:** The core part for **FastWSI** is the step of sliding window of testing block. You can refer to [s6_predWSI.py](https://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/codes/s6_predWSI.py) for details, function `fast_wsi_pred`.  


## Result
The valuable thing is that we achieve a fast WSI prediction method, not improve accuracy or performance.
We show one WSI prediction probability map below, **it only takes about 4s**.  
![wsi](://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/results/wsi.png)
![fastwsiresult](https://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/results/FastWSI_Pred.png)

