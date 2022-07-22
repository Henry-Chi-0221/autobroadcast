import os 
from glob import glob

#print(sorted(glob('results/*/*')))
path = '../videos/*/*'
for i in sorted(glob(path)):
    if i.split('/')[-1] == "out_L.mp4":
        new_L = os.path.join(i.split('/')[0] , i.split('/')[1] , "out_L_old.mp4")
        os.rename(i , new_L)
        
    if i.split('/')[-1] == "out_R.mp4":
        new_R = os.path.join(i.split('/')[0] , i.split('/')[1] , "out_L.mp4")
        os.rename(i , new_R)
for i in sorted(glob(path)):
     if i.split('/')[-1] == "out_L_old.mp4":
        new_L = os.path.join(i.split('/')[0] , i.split('/')[1] , "out_R.mp4")
        os.rename(i , new_L)
