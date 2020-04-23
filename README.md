# Fast WSI Prediction  
**Fast WSI prediction by ScanNet in PyTorch.**  


## Aim: undergraduate course  
**Invasive Ductal Carcinoma (IDC) detection** on Whole Slide Image (WSI) by deep learning method eqipped with **[ScanNet](https://ieeexplore.ieee.org/abstract/document/8354169/) fully conv. scheme** for fast WSI prediction.  

  
## Data  
**279 slides** with IDC ROI annotations.  
Ratio of **6:2:2** for training, validation and testing separation.  

Tiling small patch with 50\*50 at 2.5x magnification for `non-IDC` and `IDC` class, binary classification task.  

| patches distribution     |  non-IDC  |  IDC  |
| -------------------------|  -----    | ----- |
| training                 | 111090    | 49204 |
| validation               | 45356     | 15354 |
| testing                  | 42292     | 14228 |


## Method
1. Replace the original `VGG` with `VGG_fullyConv` for training  
2. Train the `VGG_fullyConv` model for classification, same as original training process  
3. Check the performance on `Validation` dataset and select the best model in terms of F1 score  
4. Test and evaluate your trained `VGG_fullyConv` model on `Testing` dataset in *patch-level*   
5. Infer and predict the probability map for one WSI and show it!  

### HotSpot  
> We significantly improve the inferring time by the [ScanNet](https://ieeexplore.ieee.org/abstract/document/8354169) scheme.  

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

Also, we modify the `VGG` network due to our small training size, **and**, we **remove padding operation** in convolutional layer to avoid the *border effect arcoss testing blocks*. You can refer to [myModelVgg.py](https://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/codes/myModelVgg.py) for more details.  


**Noting:** The core part for **Fast WSI** is the step of sliding window of testing block. You can refer to [s6_predWSI.py](https://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/codes/s6_predWSI.py) for details in function `fast_wsi_pred`  


## Result
### When training
We shows the curves of training loss and validation accuracy, see below:
<table border=0 width="30px" >
	<tbody> 
    <tr>		<td width="30%" align="center"> Training loss curve </td>
			<td width="30%" align="center"> Validation accuracy </td>
		</tr>
		<tr>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/results/s7_plotMetrics_Loss.png"> </td>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/results/s7_plotMetrics_Accuracy.png"> </td>
		</tr>
	</tbody>
</table>

### Independent testing evaluation on Patch-level 
|==========|===========|===========|==========|
|Confusion |  predict  |           |          |
|Matrics   |  Postive  |  Negtive  |          | 
|==========|===========|===========|==========| 
|  Postive |  11041    |  3187     |  =  14228|
|  Negtive |  3391     |  38901    |  =  42292|
|==========|===========|===========|==========|
   
- Accuracy :  0.8836
- Specificity :  0.9198
- Recall :  0.7760
- Precision :  0.7650
- F1Score :  0.7705

### 
The valuable thing is that we achieve **a fast WSI prediction method**, not improve accuracy or performance.  
We show one WSI prediction probability map below  
 **it only takes about 4s!**  
<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="40%" align="center"> Original WSI Image </td>
			<td width="40%" align="center"> Predicted Prob. Maps </td>
		</tr>
		<tr>
			<td width="40%" align="center"> <img src="https://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/results/wsi.png"> </td>
			<td width="40%" align="center"> <img src="https://github.com/gatsby2016/Fast-WSI-Prediction/blob/master/results/FastWSI_Pred.png"> </td>
		</tr>
	</tbody>
</table>
