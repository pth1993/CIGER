from .data_utils import *

seed = 343
np.random.seed(seed=seed)
random.seed(a=seed)
torch.manual_seed(seed)


class DataReader(object):
    def __init__(self, drug_file, pert_id_file, gene_file, data_file, fp_type, device, fold):
        self.device = device
        self.fp_type = fp_type
        if fp_type == 'ecfp':
            self.drug, self.drug_dim = read_drug_number(drug_file, 1024)
        elif fp_type == 'neural':
            self.drug, self.drug_dim = read_drug_string(drug_file)
        else:
            raise ValueError("Unknown fingerprint: %s" % fp_type)
        self.gene = read_gene(gene_file, 1107, self.device)

        train_pert_id, dev_pert_id, test_pert_id = split_data_by_pert_id_cv(pert_id_file, fold)
        feature_train, feature_dev, feature_test, label_train, label_dev, label_test, pos_label_train, pos_label_dev, \
        pos_label_test, neg_label_train, neg_label_dev, neg_label_test, cell_idx = \
            read_data(data_file, train_pert_id, dev_pert_id, test_pert_id, self.drug)
        self.train_feature, self.dev_feature, self.test_feature, self.train_label, self.dev_label, self.test_label, \
        self.use_pert_type, self.use_cell_id, self.use_pert_idose, self.pert_type_dim, self.cell_id_dim, \
        self.pert_idose_dim = \
            transfrom_to_tensor(feature_train, feature_dev, feature_test, label_train, label_dev, label_test, 
                                pos_label_train, pos_label_dev, pos_label_test, neg_label_train, neg_label_dev, 
                                neg_label_test, self.drug, fp_type, self.device)
        self.cell_idx = cell_idx

    def get_batch_data(self, dataset, batch_size, shuffle):
        if dataset == 'train':
            feature = self.train_feature
            label = self.train_label
        elif dataset == 'dev':
            feature = self.dev_feature
            label = self.dev_label
        elif dataset == 'test':
            feature = self.test_feature
            label = self.test_label
        if shuffle:
            index = torch.randperm(len(feature['drug'])).long()
            if self.fp_type == 'neural':
                index = index.numpy()
            else:
                index = index.to(self.device)
        for start_idx in range(0, len(feature['drug']), batch_size):
            if shuffle:
                excerpt = index[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            output_feature = dict()
            output_label = dict()
            if self.fp_type == 'neural':
                output_feature['drug'] = convert_smile_to_feature(feature['drug'][excerpt], self.device)
                output_feature['mask'] = create_mask_feature(output_feature['drug'], self.device)
            else:
                output_feature['drug'] = feature['drug'][excerpt]
            if self.use_pert_type:
                output_feature['pert_type'] = feature['pert_type'][excerpt]
            if self.use_cell_id:
                output_feature['cell_id'] = feature['cell_id'][excerpt]
            if self.use_pert_idose:
                output_feature['pert_idose'] = feature['pert_idose'][excerpt]
            output_feature['gene'] = torch.arange(978).repeat(len(output_feature['cell_id'])).\
                reshape(len(output_feature['cell_id']), 978).to(self.device)
            output_label['full_real'] = label['full_real'][excerpt]
            output_label['full_binary_pos'] = label['full_binary_pos'][excerpt]
            output_label['full_binary_neg'] = label['full_binary_neg'][excerpt]
            yield output_feature, output_label


class DataReaderDB(object):
    def __init__(self, drug_file, gene_file, device):
        self.device = device
        drug, self.drug_dim = read_drug_string(drug_file)
        drug_id, drug_smiles = zip(*sorted(drug.items()))
        self.drug = np.array(drug_smiles)
        self.gene = read_gene(gene_file, 1107, self.device)

    def get_batch_data(self, batch_size, cell_idx):
        for start_idx in range(0, len(self.drug), batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)
            output_feature = dict()
            output_feature['drug'] = convert_smile_to_feature(self.drug[excerpt], self.device)
            output_feature['mask'] = create_mask_feature(output_feature['drug'], self.device)
            output_feature['gene'] = torch.arange(978).repeat(len(self.drug[excerpt])).\
                reshape(len(self.drug[excerpt]), 978).to(self.device)
            cell_mtx = np.zeros((len(self.drug[excerpt]), 10), dtype=np.float32)
            cell_mtx[:, cell_idx] = 1
            output_feature['cell'] = torch.from_numpy(cell_mtx).to(self.device)
            yield output_feature
