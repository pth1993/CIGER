import numpy as np
import torch
import csv
from sklearn.utils import shuffle
from .molecules import Molecules


def read_drug_number(input_file, num_feature):
    drug = []
    drug_vec = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            assert len(line) == num_feature + 1, "Wrong format"
            bin_vec = [float(i) for i in line[1:]]
            drug.append(line[0])
            drug_vec.append(bin_vec)
    drug_vec = np.asarray(drug_vec, dtype=np.float32)
    index = []
    for i in range(np.shape(drug_vec)[1]):
        if len(set(drug_vec[:, i])) > 1:
            index.append(i)
    drug_vec = drug_vec[:, index]
    drug = dict(zip(drug, drug_vec))
    return drug, len(index)


def read_drug_string(input_file):
    with open(input_file, 'r') as f:
        drug = dict()
        for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_MINIMAL):
            drug[line[0]] = line[1]
    return drug, {'atom': 62, 'bond': 6}


def read_gene(input_file, num_feature, device):
    with open(input_file, 'r') as f:
        gene = []
        for line in f:
            line = line.strip().split(',')
            assert len(line) == num_feature + 1, "Wrong format"
            gene.append([float(i) for i in line[1:]])
    return torch.from_numpy(np.asarray(gene, dtype=np.float32)).to(device)


def convert_smile_to_feature(smiles, device):
    molecules = Molecules(smiles)
    node_repr = torch.FloatTensor([node.data for node in molecules.get_node_list('atom')]).to(device)
    edge_repr = torch.FloatTensor([node.data for node in molecules.get_node_list('bond')]).to(device)
    return {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}


def create_mask_feature(data, device):
    batch_idx = data['molecules'].get_neighbor_idx_by_batch('atom')
    molecule_length = [len(idx) for idx in batch_idx]
    mask = torch.zeros(len(batch_idx), max(molecule_length)).to(device)
    for idx, length in enumerate(molecule_length):
        mask[idx][:length] = 1
    return mask


def choose_mean_example(examples):
    num_example = len(examples)
    mean_value = (num_example - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value)**2
    index = np.argmin(distance)
    return examples[index]


def split_data_by_pert_id_cv(input_file, fold):
    with open(input_file) as f:
        pert_id = f.readline().strip().split(',')
    shuffle(pert_id, random_state=87)
    num_pert_id = len(pert_id)
    fold_size = int(num_pert_id / 10)
    fold_0 = pert_id[:fold_size * 2]
    fold_1 = pert_id[fold_size * 2:fold_size * 4]
    fold_2 = pert_id[fold_size * 4:fold_size * 6]
    fold_3 = pert_id[fold_size * 6:fold_size * 8]
    fold_4 = pert_id[fold_size * 8:]
    if fold == 0:
        test_pert_id = fold_0
        dev_pert_id = fold_1
        train_pert_id = fold_2 + fold_3 + fold_4
    elif fold == 1:
        test_pert_id = fold_1
        dev_pert_id = fold_2
        train_pert_id = fold_3 + fold_4 + fold_0
    elif fold == 2:
        test_pert_id = fold_2
        dev_pert_id = fold_3
        train_pert_id = fold_4 + fold_0 + fold_1
    elif fold == 3:
        test_pert_id = fold_3
        dev_pert_id = fold_4
        train_pert_id = fold_0 + fold_1 + fold_2
    elif fold == 4:
        test_pert_id = fold_4
        dev_pert_id = fold_0
        train_pert_id = fold_1 + fold_2 + fold_3
    else:
        raise ValueError("Unknown fold number: %d" % fold)
    return train_pert_id, dev_pert_id, test_pert_id


def read_data(input_file, drug_train, drug_dev, drug_test, drug):
    feature_train = []
    label_train = []
    feature_dev = []
    label_dev = []
    feature_test = []
    label_test = []
    data = dict()
    with open(input_file, 'r') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip().split(',')
            assert len(line) == 983, "Wrong format"
            if line[1] in drug:
                ft = ','.join(line[1:5])
                lb = [float(i) for i in line[5:]]
                if ft in data.keys():
                    data[ft].append(lb)
                else:
                    data[ft] = [lb]
    for ft, lb in sorted(data.items()):
        ft = ft.split(',')
        if ft[0] in drug_train:
            feature_train.append(ft)
            if len(lb) == 1:
                label_train.append(lb[0])
            else:
                lb = choose_mean_example(lb)
                label_train.append(lb)
        elif ft[0] in drug_dev:
            feature_dev.append(ft)
            if len(lb) == 1:
                label_dev.append(lb[0])
            else:
                lb = choose_mean_example(lb)
                label_dev.append(lb)
        elif ft[0] in drug_test:
            feature_test.append(ft)
            if len(lb) == 1:
                label_test.append(lb[0])
            else:
                lb = choose_mean_example(lb)
                label_test.append(lb)
        else:
            raise ValueError('Unknown drug')
    label_train = np.array(label_train, dtype=np.float32)
    label_dev = np.array(label_dev, dtype=np.float32)
    label_test = np.array(label_test, dtype=np.float32)
    label = np.concatenate([label_train, label_dev, label_test], axis=0)
    label = np.asarray(label, dtype=np.float32)
    pos_threshold = np.quantile(label, 0.95)
    neg_threshold = np.quantile(label, 0.05)
    pos_label_train = np.asarray((label_train > pos_threshold) * 1.0, dtype=np.float32)
    neg_label_train = np.asarray((label_train < neg_threshold) * 1.0, dtype=np.float32)
    pos_label_dev = np.asarray((label_dev > pos_threshold) * 1.0, dtype=np.float32)
    neg_label_dev = np.asarray((label_dev < neg_threshold) * 1.0, dtype=np.float32)
    pos_label_test = np.asarray((label_test > pos_threshold) * 1.0, dtype=np.float32)
    neg_label_test = np.asarray((label_test < neg_threshold) * 1.0, dtype=np.float32)

    tmp_data = np.asarray(feature_test)
    cell_list = ['A375', 'A549', 'HA1E', 'HCC515', 'HELA', 'HT29', 'MCF7', 'PC3', 'VCAP', 'YAPC']
    cell_idx = []
    for c in cell_list:
        c_idx = tmp_data[:, 2] == c
        cell_idx.append(c_idx)

    return np.asarray(feature_train), np.asarray(feature_dev), np.asarray(feature_test), np.asarray(label_train), \
           np.asarray(label_dev), np.asarray(label_test), pos_label_train, pos_label_dev, pos_label_test, \
           neg_label_train, neg_label_dev, neg_label_test, cell_idx


def transfrom_to_tensor(feature_train, feature_dev, feature_test, label_train, label_dev, label_test, pos_label_train, 
                        pos_label_dev, pos_label_test, neg_label_train, neg_label_dev, neg_label_test, drug, fp_type,
                        device):
    train_drug_feature = []
    dev_drug_feature = []
    test_drug_feature = []
    pert_type_set = sorted(list(set(feature_train[:, 1])))
    cell_id_set = sorted(list(set(feature_train[:, 2])))
    pert_idose_set = sorted(list(set(feature_train[:, 3])))
    use_pert_type = False
    use_cell_id = False
    use_pert_idose = False
    if len(pert_type_set) > 1:
        pert_type_dict = dict(zip(pert_type_set, list(range(len(pert_type_set)))))
        train_pert_type_feature = []
        dev_pert_type_feature = []
        test_pert_type_feature = []
        use_pert_type = True
    if len(cell_id_set) > 1:
        cell_id_dict = dict(zip(cell_id_set, list(range(len(cell_id_set)))))
        train_cell_id_feature = []
        dev_cell_id_feature = []
        test_cell_id_feature = []
        use_cell_id = True
    if len(pert_idose_set) > 1:
        pert_idose_dict = dict(zip(pert_idose_set, list(range(len(pert_idose_set)))))
        train_pert_idose_feature = []
        dev_pert_idose_feature = []
        test_pert_idose_feature = []
        use_pert_idose = True
    print('Feature Summary:')
    print(pert_type_set)
    print(cell_id_set)
    print(pert_idose_set)

    for i, ft in enumerate(feature_train):
        drug_fp = drug[ft[0]]
        train_drug_feature.append(drug_fp)
        if use_pert_type:
            pert_type_feature = np.zeros(len(pert_type_set))
            pert_type_feature[pert_type_dict[ft[1]]] = 1
            train_pert_type_feature.append(np.array(pert_type_feature, dtype=np.float32))
        if use_cell_id:
            cell_id_feature = np.zeros(len(cell_id_set))
            cell_id_feature[cell_id_dict[ft[2]]] = 1
            train_cell_id_feature.append(np.array(cell_id_feature, dtype=np.float32))
        if use_pert_idose:
            pert_idose_feature = np.zeros(len(pert_idose_set))
            pert_idose_feature[pert_idose_dict[ft[3]]] = 1
            train_pert_idose_feature.append(np.array(pert_idose_feature, dtype=np.float32))

    for i, ft in enumerate(feature_dev):
        drug_fp = drug[ft[0]]
        dev_drug_feature.append(drug_fp)
        if use_pert_type:
            pert_type_feature = np.zeros(len(pert_type_set))
            pert_type_feature[pert_type_dict[ft[1]]] = 1
            dev_pert_type_feature.append(np.array(pert_type_feature, dtype=np.float32))
        if use_cell_id:
            cell_id_feature = np.zeros(len(cell_id_set))
            cell_id_feature[cell_id_dict[ft[2]]] = 1
            dev_cell_id_feature.append(np.array(cell_id_feature, dtype=np.float32))
        if use_pert_idose:
            pert_idose_feature = np.zeros(len(pert_idose_set))
            pert_idose_feature[pert_idose_dict[ft[3]]] = 1
            dev_pert_idose_feature.append(np.array(pert_idose_feature, dtype=np.float32))

    for i, ft in enumerate(feature_test):
        drug_fp = drug[ft[0]]
        test_drug_feature.append(drug_fp)
        if use_pert_type:
            pert_type_feature = np.zeros(len(pert_type_set))
            pert_type_feature[pert_type_dict[ft[1]]] = 1
            test_pert_type_feature.append(np.array(pert_type_feature, dtype=np.float32))
        if use_cell_id:
            cell_id_feature = np.zeros(len(cell_id_set))
            cell_id_feature[cell_id_dict[ft[2]]] = 1
            test_cell_id_feature.append(np.array(cell_id_feature, dtype=np.float32))
        if use_pert_idose:
            pert_idose_feature = np.zeros(len(pert_idose_set))
            pert_idose_feature[pert_idose_dict[ft[3]]] = 1
            test_pert_idose_feature.append(np.array(pert_idose_feature, dtype=np.float32))

    train_feature = dict()
    dev_feature = dict()
    test_feature = dict()
    train_label = dict()
    dev_label = dict()
    test_label = dict()
    if fp_type == 'ecfp':
        train_feature['drug'] = torch.from_numpy(np.asarray(train_drug_feature, dtype=np.float32)).to(device)
        dev_feature['drug'] = torch.from_numpy(np.asarray(dev_drug_feature, dtype=np.float32)).to(device)
        test_feature['drug'] = torch.from_numpy(np.asarray(test_drug_feature, dtype=np.float32)).to(device)
    elif fp_type == 'neural':
        train_feature['drug'] = np.asarray(train_drug_feature)
        dev_feature['drug'] = np.asarray(dev_drug_feature)
        test_feature['drug'] = np.asarray(test_drug_feature)
    if use_pert_type:
        train_feature['pert_type'] = torch.from_numpy(np.asarray(train_pert_type_feature, dtype=np.float32)).to(device)
        dev_feature['pert_type'] = torch.from_numpy(np.asarray(dev_pert_type_feature, dtype=np.float32)).to(device)
        test_feature['pert_type'] = torch.from_numpy(np.asarray(test_pert_type_feature, dtype=np.float32)).to(device)
    if use_cell_id:
        train_feature['cell_id'] = torch.from_numpy(np.asarray(train_cell_id_feature, dtype=np.float32)).to(device)
        dev_feature['cell_id'] = torch.from_numpy(np.asarray(dev_cell_id_feature, dtype=np.float32)).to(device)
        test_feature['cell_id'] = torch.from_numpy(np.asarray(test_cell_id_feature, dtype=np.float32)).to(device)
    if use_pert_idose:
        train_feature['pert_idose'] = torch.from_numpy(np.asarray(train_pert_idose_feature, dtype=np.float32)).to(device)
        dev_feature['pert_idose'] = torch.from_numpy(np.asarray(dev_pert_idose_feature, dtype=np.float32)).to(device)
        test_feature['pert_idose'] = torch.from_numpy(np.asarray(test_pert_idose_feature, dtype=np.float32)).to(device)
    
    train_label['real'] = torch.from_numpy(label_train).to(device)
    dev_label['real'] = torch.from_numpy(label_dev).to(device)
    test_label['real'] = torch.from_numpy(label_test).to(device)
    train_label['binary'] = torch.from_numpy(pos_label_train).to(device)
    dev_label['binary'] = torch.from_numpy(pos_label_dev).to(device)
    test_label['binary'] = torch.from_numpy(pos_label_test).to(device)
    train_label['binary_reverse'] = torch.from_numpy(neg_label_train).to(device)
    dev_label['binary_reverse'] = torch.from_numpy(neg_label_dev).to(device)
    test_label['binary_reverse'] = torch.from_numpy(neg_label_test).to(device)
    return train_feature, dev_feature, test_feature, train_label, dev_label, test_label, use_pert_type, use_cell_id, \
           use_pert_idose, len(pert_type_set), len(cell_id_set), len(pert_idose_set)
