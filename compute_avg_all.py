
import numpy as np
import os
def read(root, root2020):

    iteration = list(range(0 ,52 ,2))

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
            WT.append(np.float(wt_strs[ i +1].split(',')[0]))
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
    print("ET  ", ET[mi] )
    print("TC  ", TC[mi])

    path = root2020 + str(iteration[mi])
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
        return np.float(wt_strs[i + 1].split(',')[0]), np.float(et_strs[i + 1].split(',')[0]), np.float(tc_strs[i + 1].split(',')[0])





import numpy as np
import os
def avg_all(root, root2020, seed):

    iteration = list(range(0 ,52 ,2))

    WT2019 = []
    ET2019 = []
    TC2019 = []
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
                continue
            WT2019.append(np.float(wt_strs[ i +1].split(',')[0]))
            ET2019.append(np.float(et_strs[i + 1].split(',')[0]))
            TC2019.append(np.float(tc_strs[i + 1].split(',')[0]))





    iteration = list(range(0 ,52 ,2))
   # if seed == 41:  
    iteration[-1] = 49

    WT2020 = []
    ET2020 = []
    TC2020 = []
    for ind in iteration:

        path = root2020 + str(ind)
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
                continue
            WT2020.append(np.float(wt_strs[ i +1].split(',')[0]))
            ET2020.append(np.float(et_strs[i + 1].split(',')[0]))
            TC2020.append(np.float(tc_strs[i + 1].split(',')[0]))

    WT = (np.array(WT2019) * 50 + np.array(WT2020) * 34) / (34 + 50)
    ET = (np.array(ET2019) * 50 + np.array(ET2020) * 34) / (34 + 50)
    TC = (np.array(TC2019) * 50 + np.array(TC2020)* 34) / (34 + 50)

    print("WT is ", WT)
    print("ET is ", ET)
    print("TC is ", TC)
    
    return WT, ET , TC



def partially_sup_5():
    #sup is semi  partially_sup is partially_sup
    mode = 'partiallySup'


    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 5
    date_time ='2022-02-17 10:10:43.418844'
    root2019 = '/projects/sina/W-Net/cvpr2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed,date_time, year2019)
    root2020 = '/projects/sina/W-Net/cvpr2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed,date_time, year2020)


    WT1, ET1 , TC1 = avg_all(root2019, root2020, seed)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio=5
    date_time ='2022-02-17 14:06:15.057081'
    root2019 = '/projects/sina/W-Net/cvpr2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed,date_time, year2019)
    root2020 = '/projects/sina/W-Net/cvpr2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed,date_time, year2020)


    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    WT2, ET2 , TC2 = avg_all(root2019, root2020,seed)


    seed = 43

    year2019 = 'test2019'
    year2020 = 'test2020'

    date_time = '2022-02-17 16:31:21.376245'


    root2019 = '/projects/sina/W-Net/cvpr2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed,date_time, year2019)
    root2020 = '/projects/sina/W-Net/cvpr2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed,date_time, year2020)


    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    WT3, ET3 , TC3 = avg_all(root2019, root2020, seed)



    print("FINAL AVG")

    print("** WT** ")
    print( ( (WT1 + WT2 + WT3)/3) [16])
    print(np.max((WT1 + WT2 + WT3)/3))

    print("** ET **")
    print( ((ET1 + ET2 + ET3)/3) [16])
    print("***TC***")
    print( ((TC1 + TC2 + TC3)/3) [16])


def semi_sup_5():
    # sup is semi  partially_sup is partially_sup
    mode = 'partiallySup'

    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 5
    date_time = '2022-02-17 10:24:29.947448'
    root2019 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)



    WT1, ET1, TC1 = avg_all(root2019, root2020, seed)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 5
    date_time = '2022-02-17 13:22:13.958454'
    root2019 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)



    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    WT2, ET2, TC2 = avg_all(root2019, root2020, seed)

    seed = 43

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 5
    date_time = '2022-02-17 16:12:50.231484'
    root2019 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)


    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    WT3, ET3, TC3 = avg_all(root2019, root2020, seed)

    print("FINAL AVG")

    print("** WT** ")
    print(((WT1 + WT2 + WT3) / 3)[16])
    print(np.max((WT1 + WT2 + WT3) / 3))

    print("** ET **")
    print(((ET1 + ET2 + ET3) / 3)[16])
    print("***TC***")
    print(((TC1 + TC2 + TC3) / 3)[16])



def partially_sup_3():
    #sup is semi  partially_sup is partially_sup
    mode = 'partiallySup'


    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-15 13:25:46.112535'
    root2019 = '/projects/sina/W-Net/cvpr2022/partiallySup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed, date_time, year2019)
    root2020 = '/projects/sina/W-Net/cvpr2022/partiallySup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed, date_time, year2020)

    WT1, ET1 , TC1 = avg_all(root2019, root2020, seed)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio=3
    date_time = '2022-02-16 10:36:06.734794'

    root2019 = '/projects/sina/W-Net/cvpr2022/partiallySup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed, date_time, year2019)
    root2020 = '/projects/sina/W-Net/cvpr2022/partiallySup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed, date_time, year2020)

    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    WT2, ET2 , TC2 = avg_all(root2019, root2020,seed)


    seed = 43

    year2019 = 'test2019'
    year2020 = 'test2020'

    date_time = '2022-02-16 07:40:06.416577'
    root2019 = '/projects/sina/W-Net/cvpr2022/partiallySup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed, date_time, year2019)
    root2020 = '/projects/sina/W-Net/cvpr2022/partiallySup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed, date_time, year2020)


    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    WT3, ET3 , TC3 = avg_all(root2019, root2020, seed)



    print("FINAL AVG")

    print("** WT** ")
    print( ( (WT1 + WT2 + WT3)/3) [16])
    print(np.max((WT1 + WT2 + WT3)/3))

    print("** ET **")
    print( ((ET1 + ET2 + ET3)/3) [16])
    print("***TC***")
    print( ((TC1 + TC2 + TC3)/3) [16])


def semi_sup_3():
    # sup is semi  partially_sup is partially_sup
    mode = 'partiallySup'

    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-15 22:14:39.973107'
    root2019 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2019)

    root2020 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2020)



    WT1, ET1, TC1 = avg_all(root2019, root2020, seed)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-16 10:28:39.015293'
    root2019 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)



    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    WT2, ET2, TC2 = avg_all(root2019, root2020, seed)

    seed = 43

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-16 07:48:53.989469'
    root2019 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/cvpr2022/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)

    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    WT3, ET3, TC3 = avg_all(root2019, root2020, seed)

    print("FINAL AVG")

    print("** WT** ")
    print(((WT1 + WT2 + WT3) / 3)[16])
    print(np.max((WT1 + WT2 + WT3) / 3))

    print("** ET **")
    print(((ET1 + ET2 + ET3) / 3)[16])
    print("***TC***")
    print(((TC1 + TC2 + TC3) / 3)[16])



# sup 3 ratio
print("superivsed")
partially_sup_3()
print('unsupervised')
semi_sup_3()


# # sup 5 ratio
# print("superivsed")
# partially_sup_5()
# print('unsupervised')
# semi_sup_5()








