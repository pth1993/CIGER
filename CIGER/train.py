import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import random

seed = 343
np.random.seed(seed=seed)
random.seed(a=seed)
torch.manual_seed(seed)

from datetime import datetime
import argparse
import math
from tqdm import tqdm
from models import CIGER
from utils import DataReader
from utils import auroc, auprc, precision_k, ndcg, kendall_tau, mean_average_precision

start_time = datetime.now()

parser = argparse.ArgumentParser(description='DeepCOP Training')
parser.add_argument('--drug_file', help='drug feature file (ECFP or SMILES)')
parser.add_argument('--drug_id_file', help='drug id file')
parser.add_argument('--gene_file', help='gene feature file')
parser.add_argument('--data_file', help='chemical signature file')
parser.add_argument('--fp_type', help='ECFP or Neural FP')
parser.add_argument('--label_type', help='real/real reverse/binary/binary reverse')
parser.add_argument('--loss_type', help='pair_wise_ranknet/list_wise_listnet/list_wise_listmle/list_wise_rankcosine/'
                                        'list_wise_ndcg')
parser.add_argument('--batch_size', help='number of training example per update')
parser.add_argument('--max_epoch', help='total number of training iterations')
parser.add_argument('--lr', help='learning rate')
parser.add_argument('--fold', help='id for testing set in cross-validation setting')
parser.add_argument('--model_name', help='name of model')
parser.add_argument('--warm_start', help='training from pre-trained model')
parser.add_argument('--inference', help='inference from pre-trained model')

args = parser.parse_args()

drug_file = args.drug_file
drug_id_file = args.drug_id_file
gene_file = args.gene_file
data_file = args.data_file
fp_type = args.fp_type
label_type = args.label_type
loss_type = args.loss_type
batch_size = int(args.batch_size)
max_epoch = int(args.max_epoch)
lr = float(args.lr)
fold = int(args.fold)
model_name = args.model_name
warm_start = True if args.warm_start == 'True' else False
inference = True if args.inference == 'True' else False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
intitializer = torch.nn.init.xavier_uniform_
num_gene = 978
data = DataReader(drug_file, drug_id_file, gene_file, data_file, fp_type, device, fold)
print('#Train: %d' % len(data.train_feature['drug']))
print('#Dev: %d' % len(data.dev_feature['drug']))
print('#Test: %d' % len(data.test_feature['drug']))

if inference:
    model = CIGER(drug_input_dim=data.drug_dim, gene_embed=data.gene, gene_input_dim=data.gene.size()[1],
                  encode_dim=512, fp_type=fp_type, loss_type=loss_type, label_type=label_type, device=device,
                  initializer=intitializer, pert_type_input_dim=data.pert_type_dim, cell_id_input_dim=data.cell_id_dim,
                  pert_idose_input_dim=data.pert_idose_dim, use_pert_type=data.use_pert_type,
                  use_cell_id=data.use_cell_id, use_pert_idose=data.use_pert_idose)
    checkpoint = torch.load('saved_model/ciger/%s_%d.ckpt' % (model_name + '_' + loss_type + '_' + label_type, fold),
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    epoch_loss = 0
    label_binary_np = np.empty([0, num_gene])
    label_real_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False))):
            ft, lb = batch
            drug = ft['drug']
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            gene = ft['gene']
            predict = model(drug, gene, pert_type, cell_id, pert_idose)
            if label_type == 'binary' or label_type == 'real':
                label = lb['binary']
            elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                label = lb['binary_reverse']
            else:
                raise ValueError('Unknown label type: %s' % label_type)

            if label_type == 'binary' or label_type == 'real':
                label_binary = lb['binary']
                label_real = lb['real']
            elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                label_binary = lb['binary_reverse']
                label_real = -lb['real']
            else:
                raise ValueError('Unknown label type: %s' % label_type)
            label_binary_np = np.concatenate((label_binary_np, label_binary.cpu().numpy()), axis=0)
            label_real_np = np.concatenate((label_real_np, label_real.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
        auroc_score = auroc(label_binary_np, predict_np)
        auprc_score = auprc(label_binary_np, predict_np)
        precision_10 = precision_k(label_real_np, predict_np, 10)
        precision_50 = precision_k(label_real_np, predict_np, 50)
        precision_100 = precision_k(label_real_np, predict_np, 100)
        precision_200 = precision_k(label_real_np, predict_np, 200)
        kendall_tau_score = kendall_tau(label_real_np, predict_np)
        map_score = mean_average_precision(label_real_np, predict_np)
        label_real_np = np.where(label_real_np < 0, 0, label_real_np)
        predict_np = np.where(predict_np < 0, 0, predict_np)
        ndcg_score = ndcg(label_real_np, predict_np)
        print('Test AUROC: %.4f' % auroc_score)
        print('Test AUPRC: %.4f' % auprc_score)
        print('Test NDCG: %.4f' % ndcg_score)
        print('Test Precision 10: %.4f' % precision_10)
        print('Test Precision 50: %.4f' % precision_50)
        print('Test Precision 100: %.4f' % precision_100)
        print('Test Precision 200: %.4f' % precision_200)
        print('Test Kendall Tau: %.4f' % kendall_tau_score)
        print('Test MAP: %.4f' % map_score)
else:
    model = CIGER(drug_input_dim=data.drug_dim, gene_embed=data.gene, gene_input_dim=data.gene.size()[1],
                  encode_dim=512, fp_type=fp_type, loss_type=loss_type, label_type=label_type, device=device,
                  initializer=intitializer, pert_type_input_dim=data.pert_type_dim, cell_id_input_dim=data.cell_id_dim,
                  pert_idose_input_dim=data.pert_idose_dim, use_pert_type=data.use_pert_type,
                  use_cell_id=data.use_cell_id, use_pert_idose=data.use_pert_idose)

    if warm_start:
        checkpoint = torch.load('saved_model/ciger/%s_%d.ckpt' % (model_name + '_' + loss_type + '_' + label_type,
                                                                  fold), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training

    best_dev_ndcg = float("-inf")
    score_list_dev = {'auroc': [], 'auprc': [], 'ndcg': [], 'p10': [], 'p50': [], 'p100': [], 'p200': []}
    score_list_test = {'auroc': [], 'auprc': [], 'ndcg': [], 'p10': [], 'p50': [], 'p100': [], 'p200': []}
    num_batch_train = math.ceil(len(data.train_feature['drug']) / batch_size)
    num_batch_dev = math.ceil(len(data.dev_feature['drug']) / batch_size)
    num_batch_test = math.ceil(len(data.test_feature['drug']) / batch_size)
    for epoch in range(max_epoch):
        print("Iteration %d:" % (epoch + 1))
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(tqdm(data.get_batch_data(dataset='train', batch_size=batch_size, shuffle=True),
                                       total=num_batch_train)):
            ft, lb = batch
            drug = ft['drug']
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            gene = ft['gene']
            if label_type == 'binary':
                label = lb['binary']
            elif label_type == 'binary_reverse':
                label = lb['binary_reverse']
            elif label_type == 'real':
                label = lb['real']
            elif label_type == 'real_reverse':
                label = -lb['real']
            else:
                raise ValueError('Unknown label type: %s' % label_type)
            optimizer.zero_grad()
            predict = model(drug, gene, pert_type, cell_id, pert_idose)
            loss = model.loss(label, predict)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Train loss:')
        print(epoch_loss / (i + 1))

        model.eval()

        epoch_loss = 0
        label_binary_np = np.empty([0, num_gene])
        label_real_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data.get_batch_data(dataset='dev', batch_size=batch_size, shuffle=False),
                                           total=num_batch_dev)):
                ft, lb = batch
                drug = ft['drug']
                if data.use_pert_type:
                    pert_type = ft['pert_type']
                else:
                    pert_type = None
                if data.use_cell_id:
                    cell_id = ft['cell_id']
                else:
                    cell_id = None
                if data.use_pert_idose:
                    pert_idose = ft['pert_idose']
                else:
                    pert_idose = None
                gene = ft['gene']
                predict = model(drug, gene, pert_type, cell_id, pert_idose)
                if label_type == 'binary' or label_type == 'real':
                    label = lb['binary']
                elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                    label = lb['binary_reverse']
                else:
                    raise ValueError('Unknown label type: %s' % label_type)

                if label_type == 'binary' or label_type == 'real':
                    label_binary = lb['binary']
                    label_real = lb['real']
                elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                    label_binary = lb['binary_reverse']
                    label_real = -lb['real']
                else:
                    raise ValueError('Unknown label type: %s' % label_type)
                label_binary_np = np.concatenate((label_binary_np, label_binary.cpu().numpy()), axis=0)
                label_real_np = np.concatenate((label_real_np, label_real.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            auroc_score = auroc(label_binary_np, predict_np)
            auprc_score = auprc(label_binary_np, predict_np)
            precision_10 = precision_k(label_real_np, predict_np, 10)
            precision_50 = precision_k(label_real_np, predict_np, 50)
            precision_100 = precision_k(label_real_np, predict_np, 100)
            precision_200 = precision_k(label_real_np, predict_np, 200)
            label_real_np = np.where(label_real_np < 0, 0, label_real_np)
            predict_np = np.where(predict_np < 0, 0, predict_np)
            ndcg_score = ndcg(label_real_np, predict_np)
            print('Dev AUROC: %.4f' % auroc_score)
            print('Dev NDCG: %.4f' % ndcg_score)
            print('Dev Precision 10: %.4f' % precision_10)
            print('Dev Precision 50: %.4f' % precision_50)
            print('Dev Precision 100: %.4f' % precision_100)
            print('Dev Precision 200: %.4f' % precision_200)
            score_list_dev['auroc'].append(auroc_score)
            score_list_dev['auprc'].append(auprc_score)
            score_list_dev['ndcg'].append(ndcg_score)
            score_list_dev['p10'].append(precision_10)
            score_list_dev['p50'].append(precision_50)
            score_list_dev['p100'].append(precision_100)
            score_list_dev['p200'].append(precision_200)

            if best_dev_ndcg < ndcg_score:
                best_dev_ndcg = ndcg_score
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           'saved_model/ciger/%s_%d.ckpt' % (model_name + '_' + loss_type + '_' + label_type, fold))

        epoch_loss = 0
        label_binary_np = np.empty([0, num_gene])
        label_real_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False),
                                           total=num_batch_test)):
                ft, lb = batch
                drug = ft['drug']
                if data.use_pert_type:
                    pert_type = ft['pert_type']
                else:
                    pert_type = None
                if data.use_cell_id:
                    cell_id = ft['cell_id']
                else:
                    cell_id = None
                if data.use_pert_idose:
                    pert_idose = ft['pert_idose']
                else:
                    pert_idose = None
                gene = ft['gene']
                predict = model(drug, gene, pert_type, cell_id, pert_idose)
                if label_type == 'binary' or label_type == 'real':
                    label = lb['binary']
                elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                    label = lb['binary_reverse']
                else:
                    raise ValueError('Unknown label type: %s' % label_type)

                if label_type == 'binary' or label_type == 'real':
                    label_binary = lb['binary']
                    label_real = lb['real']
                elif label_type == 'binary_reverse' or label_type == 'real_reverse':
                    label_binary = lb['binary_reverse']
                    label_real = -lb['real']
                else:
                    raise ValueError('Unknown label type: %s' % label_type)
                label_binary_np = np.concatenate((label_binary_np, label_binary.cpu().numpy()), axis=0)
                label_real_np = np.concatenate((label_real_np, label_real.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            auroc_score = auroc(label_binary_np, predict_np)
            auprc_score = auprc(label_binary_np, predict_np)
            precision_10 = precision_k(label_real_np, predict_np, 10)
            precision_50 = precision_k(label_real_np, predict_np, 50)
            precision_100 = precision_k(label_real_np, predict_np, 100)
            precision_200 = precision_k(label_real_np, predict_np, 200)
            label_real_np = np.where(label_real_np < 0, 0, label_real_np)
            predict_np = np.where(predict_np < 0, 0, predict_np)
            ndcg_score = ndcg(label_real_np, predict_np)
            print('Test AUROC: %.4f' % auroc_score)
            print('Test NDCG: %.4f' % ndcg_score)
            print('Test Precision 10: %.4f' % precision_10)
            print('Test Precision 50: %.4f' % precision_50)
            print('Test Precision 100: %.4f' % precision_100)
            print('Test Precision 200: %.4f' % precision_200)
            score_list_test['auroc'].append(auroc_score)
            score_list_test['auprc'].append(auprc_score)
            score_list_test['ndcg'].append(ndcg_score)
            score_list_test['p10'].append(precision_10)
            score_list_test['p50'].append(precision_50)
            score_list_test['p100'].append(precision_100)
            score_list_test['p200'].append(precision_200)

    best_dev_epoch = np.argmax(score_list_dev['auroc'])
    print("Epoch %d got best auroc on dev set: %.4f" % (best_dev_epoch + 1, score_list_dev['auroc'][best_dev_epoch]))
    print("Epoch %d got auroc on test set w.r.t dev set: %.4f" % (
    best_dev_epoch + 1, score_list_test['auroc'][best_dev_epoch]))
    best_test_epoch = np.argmax(score_list_test['auroc'])
    print("Epoch %d got auroc on test set: %.4f" % (best_test_epoch + 1, score_list_test['auroc'][best_test_epoch]))

    best_dev_epoch = np.argmax(score_list_dev['auprc'])
    print("Epoch %d got best auprc on dev set: %.4f" % (best_dev_epoch + 1, score_list_dev['auprc'][best_dev_epoch]))
    print("Epoch %d got auprc on test set w.r.t dev set: %.4f" % (
    best_dev_epoch + 1, score_list_test['auprc'][best_dev_epoch]))
    best_test_epoch = np.argmax(score_list_test['auprc'])
    print("Epoch %d got auprc on test set: %.4f" % (best_test_epoch + 1, score_list_test['auprc'][best_test_epoch]))

    best_dev_epoch = np.argmax(score_list_dev['ndcg'])
    print("Epoch %d got best ndcg on dev set: %.4f" % (best_dev_epoch + 1, score_list_dev['ndcg'][best_dev_epoch]))
    print("Epoch %d got ndcg on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, score_list_test['ndcg']
    [best_dev_epoch]))
    best_test_epoch = np.argmax(score_list_test['ndcg'])
    print("Epoch %d got ndcg on test set: %.4f" % (best_test_epoch + 1, score_list_test['ndcg'][best_test_epoch]))

    best_dev_epoch = np.argmax(score_list_dev['p10'])
    print("Epoch %d got best p10 on dev set: %.4f" % (best_dev_epoch + 1, score_list_dev['p10'][best_dev_epoch]))
    print("Epoch %d got p10 on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, score_list_test['p10']
    [best_dev_epoch]))
    best_test_epoch = np.argmax(score_list_test['p10'])
    print("Epoch %d got p10 on test set: %.4f" % (best_test_epoch + 1, score_list_test['p10'][best_test_epoch]))

    best_dev_epoch = np.argmax(score_list_dev['p50'])
    print("Epoch %d got best p50 on dev set: %.4f" % (best_dev_epoch + 1, score_list_dev['p50'][best_dev_epoch]))
    print("Epoch %d got p50 on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, score_list_test['p50']
    [best_dev_epoch]))
    best_test_epoch = np.argmax(score_list_test['p50'])
    print("Epoch %d got p50 on test set: %.4f" % (best_test_epoch + 1, score_list_test['p50'][best_test_epoch]))

    best_dev_epoch = np.argmax(score_list_dev['p100'])
    print("Epoch %d got best p100 on dev set: %.4f" % (best_dev_epoch + 1, score_list_dev['p100'][best_dev_epoch]))
    print("Epoch %d got p100 on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, score_list_test['p100']
    [best_dev_epoch]))
    best_test_epoch = np.argmax(score_list_test['p100'])
    print("Epoch %d got p100 on test set: %.4f" % (best_test_epoch + 1, score_list_test['p100'][best_test_epoch]))

    best_dev_epoch = np.argmax(score_list_dev['p200'])
    print("Epoch %d got best p200 on dev set: %.4f" % (best_dev_epoch + 1, score_list_dev['p200'][best_dev_epoch]))
    print("Epoch %d got p200 on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, score_list_test['p200']
    [best_dev_epoch]))
    best_test_epoch = np.argmax(score_list_test['p200'])
    print("Epoch %d got p200 on test set: %.4f" % (best_test_epoch + 1, score_list_test['p200'][best_test_epoch]))

end_time = datetime.now()
print(end_time - start_time)
