import numpy as np
from collections import Counter


def read_gsea_file(input_file):
    output = []
    with open(input_file, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            output.append([float(i) for i in line])
    return np.array(output)


def read_drug_id(input_file):
    with open(input_file, 'r') as f:
        drug_id = f.readline().strip().split(',')
    return drug_id


drug_id = read_drug_id('drugbank_drug_id.csv')

pos_score = read_gsea_file('enrichment_score_up.csv')
neg_score = read_gsea_file('enrichment_score_down.csv')
idx_score = pos_score * neg_score
idx_score = np.where(idx_score < 0, 1, 0)
combined_score = (pos_score - neg_score) / 2 * idx_score

top_sorted_idx = np.argsort(combined_score, axis=0)[-10:, :]
top_drug_idx = Counter(top_sorted_idx.flatten())
output = [['DrugBank ID', 'Number of cell lines of which drug appears in top-10 candidate']]
c1 = 'DrugBank ID'
c2 = 'Number of cell lines of which drug appears in top-10 candidate list'
print("{:<15} {:<15}".format(c1, c2))
for k, v in sorted(top_drug_idx.items(), key=lambda item: item[1], reverse=True):
    print("{:<15} {:<15}".format(drug_id[k], str(v)))
