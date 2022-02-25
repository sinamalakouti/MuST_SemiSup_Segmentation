
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
                continue
            WT.append(np.float(wt_strs[ i +1].split(',')[0]))
            ET.append(np.float(et_strs[i + 1].split(',')[0]))
            TC.append(np.float(tc_strs[i + 1].split(',')[0]))

    WT = np.array(WT)
    ET = np.array(ET)
    TC = np.array(TC)
    avg = ET + TC + WT
    avg =np.array(avg)

    avg = avg / 3
    mi = np.argmax(avg)

    mv = avg[mi]
    if iteration[mi] == 50:
        r = 49
    else:
        r= iteration[mi]
    path = root2020 + str(iteration[mi])
    path = os.path.join(path, 'result.txt')
    file1 = open(path, 'r')
    Lines = file1.readlines()


    WT_line = Lines[1]
    ET_line = Lines[2]
    TC_line = Lines[3]
    wt_strs = WT_line.split(' ')
    et_strs = ET_line.split(' ')
    tc_strs = TC_line.split(' ')


    wt_dsc = None
    et_dsc = None
    tc_dsc = None

    wt_hd = None
    et_hd = None
    tc_hd = None
    for i, s in enumerate(wt_strs):
        if s == 'DICE:':
            wt_dsc, et_dsc, tc_dsc = (np.float(wt_strs[i + 1].split(',')[0]), np.float(et_strs[i + 1].split(',')[0]), np.float(tc_strs[i + 1].split(',')[0]))
        elif s == 'hd:':
            wt_hd, et_hd, tc_hd = (np.float(wt_strs[i + 1].split(',')[0]), np.float(et_strs[i + 1].split(',')[0]),
                                      np.float(tc_strs[i + 1].split(',')[0]))
    return (wt_dsc, et_dsc, tc_dsc), (wt_hd, et_hd, tc_hd)









def partially_sup_10():
    #sup is semi  partially_sup is partially_sup
    mode = 'partiallySup'


    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 10
    date_time ='2022-02-21 09:56:12.009571'
    root2019 = '/projects/sina/W-Net/miccai2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed,date_time, year2019)
    root2020 = '/projects/sina/W-Net/miccai2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed,date_time, year2020)

    DSC_41, HD_41 = read(root2019, root2020)


    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 10
    date_time = '2022-02-21 12:46:07.095553'
    root2019 = '/projects/sina/W-Net/miccai2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed, date_time, year2019)
    root2020 = '/projects/sina/W-Net/miccai2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed, date_time, year2020)


    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_42, HD_42 = read(root2019, root2020)


    seed = 43

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 10
    date_time = '2022-02-21 15:31:59.339258'
    root2019 = '/projects/sina/W-Net/miccai2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed, date_time, year2019)
    root2020 = '/projects/sina/W-Net/miccai2022/partially_sup/sup_ratio_{}/seed_{}/{}/result_images/{}_new_results_iter'.format(
        ratio, seed, date_time, year2020)


    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_43, HD_43 = read(root2019, root2020)


    print("FINAL AVG  DSC")

    print("** WT** ")
    print( ( DSC_41[0] + DSC_42[0]+ DSC_43[0]))

    print("** ET **")
    print( ( DSC_41[1] + DSC_42[1]+ DSC_43[1]))
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]))

    print("FINAL AVG  HD")

    print("** WT** ")
    print( ( HD_41[0] + HD_42[0]+ HD_43[0]))

    print("** ET **")
    print( ( HD_41[1] + HD_42[1]+ HD_43[1]))
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]))




def semi_sup_10():
    # sup is semi  partially_sup is partially_sup
    mode = 'partiallySup'

    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 10
    date_time = '2022-02-21 09:55:59.117838'
    root2019 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)



    DSC_41, HD_41 = read(root2019, root2020)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 10
    date_time = '2022-02-21 13:46:42.254403'
    root2019 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)



    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_42, HD_42 = read(root2019, root2020)

    seed = 43

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 10
    date_time = '2022-02-21 17:23:56.694623'
    root2019 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022/semi_alternate/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)


    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_43, HD_43 = read(root2019, root2020)



    print("FINAL AVG  DSC")

    print("** WT** ")
    print( ( DSC_41[0] + DSC_42[0]+ DSC_43[0]))

    print("** ET **")
    print( ( DSC_41[1] + DSC_42[1]+ DSC_43[1]))
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]))

    print("FINAL AVG  HD")

    print("** WT** ")
    print( ( HD_41[0] + HD_42[0]+ HD_43[0]))

    print("** ET **")
    print( ( HD_41[1] + HD_42[1]+ HD_43[1]))
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]))







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


    DSC_41, HD_41 = read(root2019, root2020)

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

    DSC_42, HD_42 = read(root2019, root2020)


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
    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    print((DSC_41[0] + DSC_42[0] + DSC_43[0]))

    print("** ET **")
    print((DSC_41[1] + DSC_42[1] + DSC_43[1]))
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]))

    print("FINAL AVG  HD")

    print("** WT** ")
    print((HD_41[0] + HD_42[0] + HD_43[0]))

    print("** ET **")
    print((HD_41[1] + HD_42[1] + HD_43[1]))
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]))


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



    DSC_41, HD_41 = read(root2019, root2020)

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

    DSC_42, HD_42 = read(root2019, root2020)

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

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    print((DSC_41[0] + DSC_42[0] + DSC_43[0]))

    print("** ET **")
    print((DSC_41[1] + DSC_42[1] + DSC_43[1]))
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]))

    print("FINAL AVG  HD")

    print("** WT** ")
    print((HD_41[0] + HD_42[0] + HD_43[0]))

    print("** ET **")
    print((HD_41[1] + HD_42[1] + HD_43[1]))
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]))


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

    DSC_41, HD_41 = read(root2019, root2020)

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

    DSC_42, HD_42 = read(root2019, root2020)


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

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    print((DSC_41[0] + DSC_42[0] + DSC_43[0]))

    print("** ET **")
    print((DSC_41[1] + DSC_42[1] + DSC_43[1]))
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]))

    print("FINAL AVG  HD")

    print("** WT** ")
    print((HD_41[0] + HD_42[0] + HD_43[0]))

    print("** ET **")
    print((HD_41[1] + HD_42[1] + HD_43[1]))
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]))


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



    DSC_41, HD_41 = read(root2019, root2020)

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

    DSC_42, HD_42 = read(root2019, root2020)

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

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    print((DSC_41[0] + DSC_42[0] + DSC_43[0]))

    print("** ET **")
    print((DSC_41[1] + DSC_42[1] + DSC_43[1]))
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]))

    print("FINAL AVG  HD")

    print("** WT** ")
    print((HD_41[0] + HD_42[0] + HD_43[0]))

    print("** ET **")
    print((HD_41[1] + HD_42[1] + HD_43[1]))
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]))


print("******  SUP 3 RATIO ******")
# sup 3 ratio
print("superivsed")
partially_sup_3()

print('semi')
semi_sup_3()




print("******  SUP 5 RATIO ******")



# print("superivsed")
partially_sup_5()
# print('unsupervised')
semi_sup_5()

print("******  SUP 10 RATIO  RESULT ******")



# print("superivsed")
partially_sup_10()
# print('unsupervised')
semi_sup_10()



