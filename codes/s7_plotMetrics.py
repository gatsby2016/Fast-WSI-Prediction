import matplotlib.pyplot as plt 
import re
import os

pathname = '../results/' + str(os.path.basename(__file__).split('.')[0])

losssavename = pathname + '_Loss.png'
accsavename = pathname + '_Accuracy.png'

filename = '../results/log_FastWSI_VGG.log'
f = open(filename,'r')
content = f.read()
f.close()
print('Load log file success!')


# pattern = re.compile('\*\*\*\*\*\*\*\*_Loss_\*\*\*\*\*\*\*\* \d*.\d*')
accpattern = re.compile('Accuracy :  \d*\.\d\d\d\d')
acc = [float(one.split(' ')[-1]) for one in accpattern.findall(content)]
#acc = acc[0::2]
plt.figure(figsize=(10, 5))
plt.plot(range(1,len(acc)+1), acc, 'r-', lw=2)
plt.xlim([1, len(acc)])
plt.ylim([0.75, 1.0])
plt.xlabel('Epoches')
plt.ylabel('Accuracy values on validation dataset')
# plt.title('Accuracy values range in Epoches for Altered / No altered')
# plt.legend(loc="lower right")
plt.savefig(accsavename, dpi=300, quality=75)
# plt.show()
print('Accuracy.png has been saved!')


losspattern = re.compile('Avgloss\d*\.\d\d\d\d')
loss = [float(one.split('s')[-1]) for one in losspattern.findall(content)]

plt.figure(figsize=(10, 5))
plt.plot(range(1,len(loss)+1), loss, 'b-.', lw=2)
plt.xlim([1, len(loss)])
plt.ylim([min(loss), max(loss)])
plt.xlabel('Epoches')
plt.ylabel('Loss values range on training dataset')
# plt.title('Loss & Accuracy values range in Epoches for Altered / No altered')
# plt.legend(loc="lower right")
plt.savefig(losssavename, dpi=300, quality=75)
# plt.show()
print('Loss.png has been saved!')
