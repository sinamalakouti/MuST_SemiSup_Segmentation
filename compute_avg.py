
import numpy as np
import os
def read(root):

    iteration = list(range(0,52,2))

    WT = []
    ET = []
    TC = []
    for ind in iteration: 
        
        path = root + str(ind)
        path = os.path.join(path, 'result.txt')
        file1 = open(path, 'r')
        Lines = file1.readlines()
        WT_line = Lines[1]
        ET_line = Lines[2]
        TC_line = Lines[3]

        wt_strs = WT_line.split(' ')
        et_strs = ET_line.split(' ')
        tc_strs = TC_line.split(' ')
        for i,  s in enumerate(wt_strs):
            if s != 'DICE:':
                print("s is ", s)
                continue
            WT.append(np.float(wt_strs[i+1].split(',')[0]))
            ET.append(np.float(et_strs[i + 1].split(',')[0]))
            TC.append(np.float(tc_strs[i + 1].split(',')[0]))


    print(len(ET))
    print("WT is ", WT)
    print("ET is ", ET)
    print("TC is ", TC)
    WT = np.array(WT)
    ET = np.array(ET)
    TC = np.array(TC)
    avg = ET + TC + WT
    print("avg")
    print(len(avg))
    avg =np.array(avg)
    print( "avg is ",  avg / 3)
    print('max is')
    avg = avg  /3
    mi = np.argmax(avg)
    mv = avg[mi]
    print(mv)
    print(iteration[mi])

seed = 42
ratio = 3
year = 'test2019'
root = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/sup_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(ratio, seed, year)

read(root)
