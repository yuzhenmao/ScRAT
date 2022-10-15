import numpy as np
import pickle


def Covid_data(args):
    if args.pca == True:
        with open('covid_pca.npy', 'rb') as f:
            origin = np.load(f)
    else:
        with open('origin.npy', 'rb') as f:
            origin = np.load(f)

    a_file = open('patient_id.pkl', "rb")
    patient_id = pickle.load(a_file)
    a_file.close()

    a_file = open(args.task+'_label.pkl', "rb")
    labels = pickle.load(a_file)
    a_file.close()

    a_file = open('cell_type.pkl', "rb")
    cell_type = pickle.load(a_file)
    a_file.close()

    a_file = open('cell_type_64.pkl', "rb")
    cell_type_64 = pickle.load(a_file)
    a_file.close()

    if args.task == 'severity':
        a_file = open('stage_label.pkl', "rb")
        stage_labels = pickle.load(a_file)
        a_file.close()

    id_dict = {}
    if args.task == 'severity':
        id_dict = {'mild/moderate': 0, 'severe/critical': 1, 'control': -1}
    elif args.task == 'sex':
        id_dict = {'M': 0, 'F': 1}
    elif args.task == 'stage':
        id_dict = {'convalescence': 0, 'progression': 1, 'control': -1}

    labels_ = np.array(labels.map(id_dict))

    if args.task == 'severity':
        id_dict_ = {'convalescence': 0, 'progression': 1, 'control': 0}
        labels_stage = np.array(stage_labels.map(id_dict_))

    l_dict = {}
    indices = np.arange(origin.shape[0])
    p_ids = sorted(set(patient_id))
    p_idx = []
    for i in p_ids:
        idx = indices[patient_id == i]
        if len(idx) < 500:
            continue
        if len(set(labels_[idx])) > 1:
            for ii in sorted(set(labels_[idx])):
                if ii > -1:
                    iidx = idx[labels_[idx] == ii]
                    p_idx.append(iidx)
                    l_dict[labels_[iidx[0]]] = l_dict.get(labels_[iidx[0]], 0) + 1
        else:
            if args.task == 'severity':
                if (labels_[idx[0]] > -1) and (labels_stage[idx[0]]) > 0:
                    p_idx.append(idx)
                    l_dict[labels_[idx[0]]] = l_dict.get(labels_[idx[0]], 0) + 1
            else:
                if labels_[idx[0]] > -1:
                    p_idx.append(idx)
                    l_dict[labels_[idx[0]]] = l_dict.get(labels_[idx[0]], 0) + 1

    print(l_dict)

    return [], p_idx, labels_, cell_type, patient_id, origin, cell_type_64