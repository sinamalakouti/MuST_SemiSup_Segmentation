
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
    path = root2020 + str(r)
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
    return (wt_dsc * 100, et_dsc* 100, tc_dsc* 100), (wt_hd* 100, et_hd* 100, tc_hd* 100)









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
    dsc_wt = [DSC_41[0], DSC_42[0], DSC_43[0]]
    print("WT:  mu:  {}    std: {}".format(np.mean(dsc_wt), np.std(dsc_wt)))

    print("** ET **")
    dsc_et = [DSC_41[1], DSC_42[1], DSC_43[1]]
    print("ET:  mu:  {}    std: {}".format(np.mean(dsc_et), np.std(dsc_et)))

    print("***TC***")

    dsc_tc = [DSC_41[2], DSC_42[2], DSC_43[2]]
    print("TC:  mu:  {}    std: {}".format(np.mean(dsc_tc), np.std(dsc_tc)))




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
    dsc_wt = [DSC_41[0], DSC_42[0], DSC_43[0]]
    print("WT:  mu:  {}    std: {}".format(np.mean(dsc_wt), np.std(dsc_wt)))

    print("** ET **")
    dsc_et = [DSC_41[1], DSC_42[1], DSC_43[1]]
    print("ET:  mu:  {}    std: {}".format(np.mean(dsc_et), np.std(dsc_et)))

    print("***TC***")

    dsc_tc = [DSC_41[2], DSC_42[2], DSC_43[2]]
    print("TC:  mu:  {}    std: {}".format(np.mean(dsc_tc), np.std(dsc_tc)))






def fully_sup():
    #sup is semi  partially_sup is partially_sup
    mode = 'partiallySup'


    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 10
    date_time ='2022-02-21 09:56:12.009571'
    root2019 = '/projects/sina/W-Net/miccai2022/fully_sup/fullySup_ratio_10/seed_{}/2022-02-22 02:14:41.044874/result_images/{}_new_results_iter'.format(
         seed, year2019)
    root2020 = 'projects/sina/W-Net/miccai2022/fully_sup/fullySup_ratio_10/seed_{}/2022-02-22 02:14:41.044874/result_images/{}_new_results_iter'.format(
        seed, year2020)

    DSC_41, HD_41 = read(root2019, root2020)


    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 10
    date_time = '2022-02-21 12:46:07.095553'
    root2019 = '/projects/sina/W-Net/miccai2022/fully_sup/fullySup_ratio_10/seed_42/2022-02-22 08:04:47.959944/result_images/{}_new_results_iter'.format(
         year2019)
    root2020 = '/projects/sina/W-Net/miccai2022/fully_sup/fullySup_ratio_10/seed_42/2022-02-22 08:04:47.959944/result_images/{}_new_results_iter'.format(
          year2020)


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
    root2019 = '/projects/sina/W-Net/miccai2022/fully_sup/fullySup_ratio_10/seed_43/2022-02-22 15:49:12.414664/result_images/{}_new_results_iter'.format(
        year2019)
    root2020 = '/projects/sina/W-Net/miccai2022/fully_sup/fullySup_ratio_10/seed_43/2022-02-22 15:49:12.414664/result_images/{}_new_results_iter'.format(
         year2020)


    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    dsc_wt = [DSC_41[0], DSC_42[0], DSC_43[0]]
    print("WT:  mu:  {}    std: {}".format(np.mean(dsc_wt), np.std(dsc_wt)))

    print("** ET **")
    dsc_et = [DSC_41[1], DSC_42[1], DSC_43[1]]
    print("ET:  mu:  {}    std: {}".format(np.mean(dsc_et), np.std(dsc_et)))

    print("***TC***")

    dsc_tc = [DSC_41[2], DSC_42[2], DSC_43[2]]
    print("TC:  mu:  {}    std: {}".format(np.mean(dsc_tc), np.std(dsc_tc)))






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
    dsc_wt = [DSC_41[0], DSC_42[0], DSC_43[0]]
    print("WT:  mu:  {}    std: {}".format(np.mean(dsc_wt), np.std(dsc_wt)))

    print("** ET **")
    dsc_et = [DSC_41[1], DSC_42[1], DSC_43[1]]
    print("ET:  mu:  {}    std: {}".format(np.mean(dsc_et), np.std(dsc_et)))

    print("***TC***")

    dsc_tc = [DSC_41[2], DSC_42[2], DSC_43[2]]
    print("TC:  mu:  {}    std: {}".format(np.mean(dsc_tc), np.std(dsc_tc)))

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
    dsc_wt = [DSC_41[0], DSC_42[0], DSC_43[0]]
    print("WT:  mu:  {}    std: {}".format(np.mean(dsc_wt), np.std(dsc_wt)))

    print("** ET **")
    dsc_et = [DSC_41[1], DSC_42[1], DSC_43[1]]
    print("ET:  mu:  {}    std: {}".format(np.mean(dsc_et), np.std(dsc_et)))

    print("***TC***")

    dsc_tc = [DSC_41[2], DSC_42[2], DSC_43[2]]
    print("TC:  mu:  {}    std: {}".format(np.mean(dsc_tc), np.std(dsc_tc)))


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
    dsc_wt = [DSC_41[0], DSC_42[0], DSC_43[0]]
    print("WT:  mu:  {}    std: {}".format(np.mean(dsc_wt), np.std(dsc_wt)))

    print("** ET **")
    dsc_et = [DSC_41[1], DSC_42[1], DSC_43[1]]
    print("ET:  mu:  {}    std: {}".format(np.mean(dsc_et), np.std(dsc_et)))

    print("***TC***")

    dsc_tc = [DSC_41[2], DSC_42[2], DSC_43[2]]
    print("TC:  mu:  {}    std: {}".format(np.mean(dsc_tc), np.std(dsc_tc)))

def semi_sup_3():
    #sup is semi  partially_sup is partially_sup
    mode = 'semi_sup'


    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3

    root2019 = '/projects/sina/W-Net/miccai2022_final/braTS/semi_alternate/sup_ratio_{}/seed_{}/1/result_images/{}_new_results_iter'.format(
        ratio, seed, year2019)
    root2020 = '/projects/sina/W-Net/miccai2022_final/braTS/semi_alternate/sup_ratio_{}/seed_{}/1/result_images/{}_new_results_iter'.format(
        ratio, seed, year2020)

    DSC_41, HD_41 = read(root2019, root2020)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio=3


    root2019 = '/projects/sina/W-Net/miccai2022_final/braTS/semi_alternate/sup_ratio_{}/seed_{}/1/result_images/{}_new_results_iter'.format(
        ratio, seed, year2019)
    root2020 = '/projects/sina/W-Net/miccai2022_final/braTS/semi_alternate/sup_ratio_{}/seed_{}/1/result_images/{}_new_results_iter'.format(
        ratio, seed, year2020)

    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_42, HD_42 = read(root2019, root2020)


    seed = 43

    year2019 = 'test2019'
    year2020 = 'test2020'


    root2019 = '/projects/sina/W-Net/miccai2022_final/braTS/semi_alternate/sup_ratio_{}/seed_{}/1/result_images/{}_new_results_iter'.format(
        ratio, seed, year2019)
    root2020 = '/projects/sina/W-Net/miccai2022_final/braTS/semi_alternate/sup_ratio_{}/seed_{}/1/result_images/{}_new_results_iter'.format(
        ratio, seed, year2020)


    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    dsc_wt = [DSC_41[0], DSC_42[0], DSC_43[0]]
    print("WT:  mu:  {}    std: {}".format(np.mean(dsc_wt), np.std(dsc_wt)))

    print("** ET **")
    dsc_et = [DSC_41[1], DSC_42[1], DSC_43[1]]
    print("ET:  mu:  {}    std: {}".format(np.mean(dsc_et), np.std(dsc_et)))

    print("***TC***")

    dsc_tc = [DSC_41[2], DSC_42[2], DSC_43[2]]
    print("TC:  mu:  {}    std: {}".format(np.mean(dsc_tc), np.std(dsc_tc)))


def semi_pgs_3Layerwise1():


    print("RESULTS FOR SEMI - LAYERWISE 1 - PGS")


    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-25 22:45:22.759127'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2020)



    DSC_41, HD_41 = read(root2019, root2020)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-26 03:27:11.556000'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
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
    date_time = '2022-02-26 02:32:54.337290'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)

    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    dsc_wt = [DSC_41[0] , DSC_42[0] , DSC_43[0]]
    print("WT:  mu:  {}    std: {}".format(np.mean(dsc_wt), np.std(dsc_wt)))


    print("** ET **")
    dsc_et = [DSC_41[1], DSC_42[1], DSC_43[1]]
    print("ET:  mu:  {}    std: {}".format(np.mean(dsc_et), np.std(dsc_et)))


    print("***TC***")

    dsc_tc = [DSC_41[2] , DSC_42[2] , DSC_43[2]]
    print("TC:  mu:  {}    std: {}".format(np.mean(dsc_tc), np.std(dsc_tc)))
    #
    # print("FINAL AVG  HD")
    #
    # print("** WT** ")
    # print((HD_41[0] + HD_42[0] + HD_43[0]) / 3)
    #
    # print("** ET **")
    # print((HD_41[1] + HD_42[1] + HD_43[1]) / 3)
    # print("***TC***")
    # print((HD_41[2] + HD_42[2] + HD_43[2]) / 3)

def semi_unet_3Layerwise1():


    print("RESULTS FOR SEMI - UNET - LAYERWISE 1")


    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-25 22:46:34.766439'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2020)



    DSC_41, HD_41 = read(root2019, root2020)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-26 13:30:21.507782'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
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
    date_time = '2022-02-26 15:44:42.020244'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise1/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)

    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    print((DSC_41[0] + DSC_42[0] + DSC_43[0]) / 3)

    print("** ET **")
    print((DSC_41[1] + DSC_42[1] + DSC_43[1]) / 3)
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]) / 3)

    print("FINAL AVG  HD")

    print("** WT** ")
    print((HD_41[0] + HD_42[0] + HD_43[0]) / 3)

    print("** ET **")
    print((HD_41[1] + HD_42[1] + HD_43[1]) / 3)
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]) / 3)




def semi_sup_3Layerwise2():

    print(" LAYERWISE2 result")

    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-26 06:07:18.691142'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2020)



    DSC_41, HD_41 = read(root2019, root2020)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-26 05:20:05.151941'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
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
    date_time = '2022-02-26 08:49:08.928590'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)

    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    print((DSC_41[0] + DSC_42[0] + DSC_43[0]) / 3)

    print("** ET **")
    print((DSC_41[1] + DSC_42[1] + DSC_43[1]) / 3)
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]) / 3)

    print("FINAL AVG  HD")

    print("** WT** ")
    print((HD_41[0] + HD_42[0] + HD_43[0]) / 3)

    print("** ET **")
    print((HD_41[1] + HD_42[1] + HD_43[1]) / 3)
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]) / 3)

def semi_unet_3Layerwise2():


    print("RESULTS FOR SEMI - UNET - LAYERWISE 2")


    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-26 18:25:21.278886'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2020)



    DSC_41, HD_41 = read(root2019, root2020)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-26 20:58:27.884484'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
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
    date_time = '2022-02-27 00:53:14.623309'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwise2/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)

    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    print((DSC_41[0] + DSC_42[0] + DSC_43[0]) / 3)

    print("** ET **")
    print((DSC_41[1] + DSC_42[1] + DSC_43[1]) / 3)
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]) / 3)

    print("FINAL AVG  HD")

    print("** WT** ")
    print((HD_41[0] + HD_42[0] + HD_43[0]) / 3)

    print("** ET **")
    print((HD_41[1] + HD_42[1] + HD_43[1]) / 3)
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]) / 3)




def semi_sup_3LayerwiseL():
    # sup is semi  partially_sup is partially_sup
    print("LAYERWISE L REsult")

    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-26 07:54:10.326390'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2020)



    DSC_41, HD_41 = read(root2019, root2020)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-26 11:27:27.383681'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
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
    date_time = '2022-02-26 10:32:34.524417'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)

    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    print((DSC_41[0] + DSC_42[0] + DSC_43[0]) / 3)

    print("** ET **")
    print((DSC_41[1] + DSC_42[1] + DSC_43[1]) / 3)
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]) / 3)

    print("FINAL AVG  HD")

    print("** WT** ")
    print((HD_41[0] + HD_42[0] + HD_43[0]) / 3)

    print("** ET **")
    print((HD_41[1] + HD_42[1] + HD_43[1]) / 3)
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]) / 3)


def semi_unet_3LayerwiseL():


    print("RESULTS FOR SEMI - UNET - LAYERWISE L")


    seed = 41
    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-27 03:33:29.149890'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed,date_time,  year2020)



    DSC_41, HD_41 = read(root2019, root2020)

    seed = 42

    year2019 = 'test2019'
    year2020 = 'test2020'

    ratio = 3
    date_time = '2022-02-27 06:11:36.495460'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
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
    date_time = '2022-02-27 08:31:26.054997'
    root2019 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2019)

    root2020 = '/projects/sina/W-Net/miccai2022_ablation/semi_alternate/layerwiseL/sup_ratio_{}/seed_{}/{}/' \
               'result_images/{}_new_results_iter'.format(ratio, seed, date_time, year2020)

    # root2019 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2019)
    #
    # root2020 = '/projects/sina/W-Net/PGS_result/Brats/CVPR2022/semi_pgs_CE_all2018/{}_ratio_{}/seed_{}/result_images/{}_new_results_iter'.format(mode,
    #     ratio, seed, year2020)

    DSC_43, HD_43 = read(root2019, root2020)

    print("FINAL AVG  DSC")

    print("** WT** ")
    print((DSC_41[0] + DSC_42[0] + DSC_43[0]) / 3)

    print("** ET **")
    print((DSC_41[1] + DSC_42[1] + DSC_43[1]) / 3)
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]) / 3)

    print("FINAL AVG  HD")

    print("** WT** ")
    print((HD_41[0] + HD_42[0] + HD_43[0]) / 3)

    print("** ET **")
    print((HD_41[1] + HD_42[1] + HD_43[1]) / 3)
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]) / 3)

def semi_sup_3LayerwiseL12():
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
    print((DSC_41[0] + DSC_42[0] + DSC_43[0]) / 3)

    print("** ET **")
    print((DSC_41[1] + DSC_42[1] + DSC_43[1]) / 3)
    print("***TC***")
    print((DSC_41[2] + DSC_42[2] + DSC_43[2]) / 3)

    print("FINAL AVG  HD")

    print("** WT** ")
    print((HD_41[0] + HD_42[0] + HD_43[0]) / 3)

    print("** ET **")
    print((HD_41[1] + HD_42[1] + HD_43[1]) / 3)
    print("***TC***")
    print((HD_41[2] + HD_42[2] + HD_43[2]) / 3)



# def compute_final_sts(DSC_wt, DSC_et, DSC_tc):
#     print("FINAL AVG  DSC")
#
#     print("** WT** ")
#     dsc_wt = [DSC_41[0], DSC_42[0], DSC_43[0]]
#     print("WT:  mu:  {}    std: {}".format(np.mean(dsc_wt), np.std(dsc_wt)))
#
#     print("** ET **")
#     dsc_et = [DSC_41[1], DSC_42[1], DSC_43[1]]
#     print("ET:  mu:  {}    std: {}".format(np.mean(dsc_et), np.std(dsc_et)))
#
#     print("***TC***")
#
#     dsc_tc = [DSC_41[2], DSC_42[2], DSC_43[2]]
#     print("TC:  mu:  {}    std: {}".format(np.mean(dsc_tc), np.std(dsc_tc)))





#
# print("******  SUP 3 RATIO ******")
# # sup 3 ratio
# print("superivsed")
# partially_sup_3()
# 
# print('semi')
# semi_sup_3()
# 
# 
# 
# 
# print("******  SUP 5 RATIO ******")
# 
# 
# 
# # print("superivsed")
# partially_sup_5()
# # print('unsupervised')
# semi_sup_5()
# 
# print("******  SUP 10 RATIO  RESULT ******")
# 
# 
# 
# # print("superivsed")
# partially_sup_10()
# # print('unsupervised')
# semi_sup_10()
# 
# 
# 
print("press 9 to exit!")
print("do you want ablation result or main result? if main press 0 else press 1")
input1 = input()
if int(input1) == 0:
    print('what ratio? do you want?')
    input2 = input()
    if int(input2) == int(3):
        print("partially supervised results:  \n ")
        partially_sup_3()
        print("-" * 50)
        print('Semi supervised result: \n')
        semi_sup_3()
    elif int(input2) == 5:
        print("result for    5")
        print("partially supervised results:  \n ")
        partially_sup_5()
        print("-" * 50)
        print('Semi supervised result: \n')
        semi_sup_5()
    elif int(input2) == int(10):
        print("result for    10")
        print("partially supervised results:  \n ")
        partially_sup_10()
        print("-" * 50)
        print('Semi supervised result: \n')
        semi_sup_10()

elif int(input1) == 1:
    print('what layer? do you want?')
    input2 = input()
    if int(input2) == '1':
        print("result for  layer  1")
        # print("partially supervised results:  \n ")
        # partially_sup_3()

        print('Semi supervised result: \n')
        semi_pgs_3Layerwise1()

    elif int(input2) == '2':
        print("result for   layer 2")
        print("-" * 50)
        print('Semi supervised result: \n')
        semi_sup_3Layerwise2()
    elif int(input2) == 'L':
        print("result for   layer L")
        print("-" * 50)
        print('Semi supervised result: \n')
        semi_sup_3LayerwiseL()


# semi_sup_3Layerwise1()

# semi_unet_3LayerwiseL()
# semi_sup_3LayerwiseL()
# print("FULLY SUP")
# print('- * 50 ')
#
# fully_sup