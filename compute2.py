import numpy as np
import os

# this computes the value of 2020 based on the best average value of 2019  for semi-supervised
def read(root, root2020):
    iteration = list(range(0, 52, 2))

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
        for i, s in enumerate(wt_strs):
            if s != 'DICE:':
                print("s is ", s)
                continue
            WT.append(np.float(wt_strs[i + 1].split(',')[0]))
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
    avg = np.array(avg)

    avg = avg / 3
    print("avg is ", avg / 3)

    mi = np.argmax(avg)
    print('max is')
    mv = avg[mi]
    print(mv)
    print("best val iter is")
    print(iteration[mi])
    print("each is")
    print("WT  ", WT[mi])
    print("ET  ", ET[mi])
    print("TC  ", TC[mi])
    if iteration[mi] == 50:
        r = 49
    else:
        r = iteration[mi]
    path = root2020 + str(r)
    path = os.path.join(path, 'result.txt')
    file1 = open(path, 'r')
    Lines = file1.readlines()
    print("2020 is")
    print(Lines)

    WT_line = Lines[1]
    ET_line = Lines[2]
    TC_line = Lines[3]
    testWT = None
    testET = None
    testTC = None
    wt_strs = WT_line.split(' ')
    et_strs = ET_line.split(' ')
    tc_strs = TC_line.split(' ')
    for i, s in enumerate(wt_strs):
        if s != 'DICE:':
            print("s is ", s)
            continue
        return np.float(wt_strs[i + 1].split(',')[0]), np.float(et_strs[i + 1].split(',')[0]), np.float(
            tc_strs[i + 1].split(',')[0])


print("************** SEED 41 **************")
ratio = 3
seed = 41
year2019 = 'test2019'
date_time = '2022-02-15 22:14:39.973107'
year2020 = 'test2020'

root2019 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
           'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2019)

root2020 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
           'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2020)
#
# root2020 = '/projects/sina/W-Net/cvpr2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
#            'result_images/{}_new_results_iter'.format(ratio, seed, date_time,year2020)

#CVPR
# root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/sup_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(
#     ratio, seed, year2019)
#
# root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/sup_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(
#     ratio, seed, year2020)

w41, e41, t41 = read(root2019, root2020)


print("************** SEED 42 **************")

seed = 42
date_time = '2022-02-16 10:28:39.015293'
year2019 = 'test2019'
year2020 = 'test2020'


root2019 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
           'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2019)

root2020 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
           'result_images/{}_new_results_iter'.format(ratio, seed, date_time,year2020)


#       CVPR
# root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/sup_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(
#     ratio, seed, year2019)
#
# root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/sup_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(
#     ratio, seed, year2020)

w42, e42, t42 = read(root2019, root2020)

print("************** SEED 43 **************")

seed = 43

year2019 = 'test2019'
year2020 = 'test2020'
date_time = '2022-02-16 07:48:53.989469'

root2019 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
           'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2019)

root2020 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
           'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)

#CVPR
# root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/sup_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(
#     ratio, seed, year2019)
#
# root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/sup_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(
#     ratio, seed, year2020)

w43, e43, t43 = read(root2019, root2020)

print("FINAL RESULT")
print("WT    ", (w41 + w42 + w43) / 3)
print("ET    ", (e41 + e42 + e43) / 3)
print("TC    ", (t41 + t42 + t43) / 3)
