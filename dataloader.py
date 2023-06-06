import numpy as np
import pickle
import scanpy


def Covid_data(args):
    if args.task == 'haniffa':
        id_dict = {'Critical ': 1, 'Death': -1, 'Severe': 1, 'nan': -1, 'LPS': 0, 'Non-covid': 0, 'Asymptomatic': 1,
                   'Mild': 1, 'Healthy': 0, 'Moderate': 1}

        if args.pca == True:
            with open('./data/Haniffa/Haniffa_X_pca.npy', 'rb') as f:
                origin = np.load(f)
        else:
            with open('./data/Haniffa/origin.npy', 'rb') as f:
                origin = np.load(f)

        a_file = open('./data/Haniffa/patient_id.pkl', "rb")
        patient_id = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/Haniffa/labels.pkl', "rb")
        labels = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/Haniffa/cell_type.pkl', "rb")
        cell_type = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/Haniffa/cell_type_large.pkl', "rb")
        cell_type_large = pickle.load(a_file)
        a_file.close()
    elif args.task == 'combat':
        id_dict = {'COVID_HCW_MILD': 1, 'COVID_CRIT': 1, 'COVID_MILD': 1, 'COVID_SEV': 1, 'COVID_LDN': 1, 'HV': 0,
                   'Flu': 0, 'Sepsis': 0}

        if args.pca == True:
            with open('./data/COMBAT/COMBAT_X_pca.npy', 'rb') as f:
                origin = np.load(f)
        else:
            with open('./data/COMBAT/origin.npy', 'rb') as f:
                origin = np.load(f)

        a_file = open('./data/COMBAT/patient_id.pkl', "rb")
        patient_id = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/COMBAT/labels.pkl', "rb")
        labels = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/COMBAT/cell_type.pkl', "rb")
        cell_type = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/COMBAT/cell_type_large.pkl', "rb")
        cell_type_large = pickle.load(a_file)
        a_file.close()
    else:
        id_dict = {}
        if args.task == 'severity':
            id_dict = {'mild/moderate': 0, 'severe/critical': 1, 'control': -1}
        elif args.task == 'stage':
            id_dict = {'convalescence': 0, 'progression': 1, 'control': -1}

        if args.pca == True:
            with open('./data/SC4/covid_pca.npy', 'rb') as f:
                origin = np.load(f)
        else:
            with open('./data/SC4/origin.npy', 'rb') as f:
                origin = np.load(f)

        a_file = open('./data/SC4/patient_id.pkl', "rb")
        patient_id = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/SC4/' + args.task + '_label.pkl', "rb")
        labels = pickle.load(a_file)
        a_file.close()

        if args.task == 'severity':
            a_file = open('./data/SC4/stage_label.pkl', "rb")
            stage_labels = pickle.load(a_file)
            a_file.close()

        a_file = open('./data/SC4/cell_type.pkl', "rb")
        cell_type = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/SC4/cell_type_large.pkl', "rb")
        cell_type_large = pickle.load(a_file)
        a_file.close()

    labels_ = np.array(labels.map(id_dict))

    if args.task == 'severity':
        id_dict_ = {'convalescence': 0, 'progression': 1, 'control': 0}
        labels_stage = np.array(stage_labels.map(id_dict_))

    l_dict = {}
    indices = np.arange(origin.shape[0])
    p_ids = sorted(set(patient_id))
    p_idx = []

    if args.task == 'combat':
        top_class = []
        for tc in (
        'PB', 'CD4.TEFF.prolif', 'PLT', 'B.INT', 'CD8.TEFF.prolif', 'B.MEM', 'NK.cyc', 'RET', 'B.NAIVE', 'NK.mitohi'):
            top_class.append(indices[cell_type_large == tc])
        selected = np.concatenate(top_class)
    elif args.task == 'haniffa':
        top_class = []
        for tc in (
        'B_immature', 'C1_CD16_mono', 'CD4.Prolif', 'HSC_erythroid', 'RBC', 'Plasma_cell_IgG', 'pDC', 'Plasma_cell_IgA',
        'Platelets', 'Plasmablast'):
            top_class.append(indices[cell_type_large == tc])
        selected = np.concatenate(top_class)
    elif args.task == 'severity':
        top_class = []
        for tc in (
        'Macro_c3-EREG', 'Epi-Squamous', 'Neu_c5-GSTP1(high)OASL(low)', 'Epi-Ciliated', 'Neu_c3-CST7', 'Neu_c4-RSAD2',
        'Epi-Secretory', 'Mega', 'Neu_c1-IL1B', 'Macro_c6-VCAN', 'DC_c3-LAMP3', 'Neu_c6-FGF23', 'Macro_c2-CCL3L1',
        'Mono_c1-CD14-CCL3', 'Neu_c2-CXCR4(low)', 'B_c05-MZB1-XBP1', 'DC_c1-CLEC9A', 'Mono_c4-CD14-CD16'):
            top_class.append(indices[cell_type_large == tc])
        selected = np.concatenate(top_class)
    elif args.task == 'stage':
        top_class = []
        for tc in (
        'Neu_c5-GSTP1(high)OASL(low)', 'Neu_c3-CST7', 'Macro_c3-EREG', 'Epi-Squamous', 'Mega', 'Epi-Ciliated',
        'Mono_c5-CD16', 'Neu_c4-RSAD2', 'Epi-Secretory', 'Neu_c1-IL1B', 'DC_c1-CLEC9A', 'DC_c3-LAMP3',
        'Neu_c2-CXCR4(low)', 'Mono_c4-CD14-CD16', 'Mono_c1-CD14-CCL3', 'Macro_c6-VCAN'):
            top_class.append(indices[cell_type_large == tc])
        selected = np.concatenate(top_class)
    for i in p_ids:
        idx = indices[patient_id == i]
        if len(idx) < 500:
            continue
        if len(set(labels_[idx])) > 1:
            for ii in sorted(set(labels_[idx])):
                if ii > -1:
                    iidx = idx[labels_[idx] == ii]
                    tt_idx = iidx
                    # tt_idx = np.intersect1d(iidx, selected)
                    # tt_idx = np.setdiff1d(iidx, selected)
                    if len(tt_idx) < 1:
                        continue
                    p_idx.append(tt_idx)
                    l_dict[labels_[iidx[0]]] = l_dict.get(labels_[iidx[0]], 0) + 1
        else:
            if args.task == 'severity':
                if (labels_[idx[0]] > -1) and (labels_stage[idx[0]]) > 0:
                    tt_idx = idx
                    # tt_idx = np.intersect1d(idx, selected)
                    # tt_idx = np.setdiff1d(idx, selected)
                    if len(tt_idx) < 1:
                        continue
                    p_idx.append(tt_idx)
                    l_dict[labels_[idx[0]]] = l_dict.get(labels_[idx[0]], 0) + 1
            else:
                if labels_[idx[0]] > -1:
                    tt_idx = idx
                    # tt_idx = np.intersect1d(idx, selected)
                    # tt_idx = np.setdiff1d(idx, selected)
                    if len(tt_idx) < 1:
                        continue
                    p_idx.append(tt_idx)
                    l_dict[labels_[idx[0]]] = l_dict.get(labels_[idx[0]], 0) + 1

    # print(l_dict)

    return p_idx, labels_, cell_type, patient_id, origin, cell_type_large


def Custom_data(args):
    '''
    !!! Need to change line 178 before running the code !!!
    '''
    id_dict = {}  # {'cancer': 1, 'health': 0}
    data = scanpy.read_h5ad(args.dataset)
    if args.pca == True:
        origin = data.obsm['X_pca']
    else:
        origin = data.layers['raw']
    
    patient_id = data.obs['patient_id']

    labels = data.obs['Outcome']

    cell_type = data.obs['cell_type']

    cell_type_large = None
    # This (high resolution) cell_type is only for attention analysis, not necessary
    # cell_type_large = data.obs['cell_type_large']

    labels_ = np.array(labels.map(id_dict))

    l_dict = {}
    indices = np.arange(origin.shape[0])
    p_ids = sorted(set(patient_id))
    p_idx = []

    for i in p_ids:
        idx = indices[patient_id == i]
        if len(set(labels_[idx])) > 1:   # one patient with more than one labels
            for ii in sorted(set(labels_[idx])):
                if ii > -1:
                    iidx = idx[labels_[idx] == ii]
                    tt_idx = iidx
                    if len(tt_idx) < 500:  # exclude the sample with the number of cells fewer than 500
                        continue
                    p_idx.append(tt_idx)
                    l_dict[labels_[iidx[0]]] = l_dict.get(labels_[iidx[0]], 0) + 1
        else:
            if labels_[idx[0]] > -1:
                tt_idx = idx
                if len(tt_idx) < 500:  # exclude the sample with the number of cells fewer than 500
                    continue
                p_idx.append(tt_idx)
                l_dict[labels_[idx[0]]] = l_dict.get(labels_[idx[0]], 0) + 1

    # print(l_dict)

    return p_idx, labels_, cell_type, patient_id, origin, cell_type_large
