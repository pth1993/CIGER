import torch
import torch.nn as nn
from .ltr_loss import point_wise_mse, list_wise_listnet, list_wise_listmle, pair_wise_ranknet, list_wise_rankcosine, \
    list_wise_ndcg
from .neural_fingerprint import NeuralFingerprint
from .attention import Attention


class CIGER(nn.Module):
    def __init__(self, drug_input_dim, gene_embed, gene_input_dim, encode_dim, fp_type, loss_type, label_type, device, initializer=None,
                 pert_type_input_dim=None, cell_id_input_dim=None, pert_idose_input_dim=None, use_pert_type=False,
                 use_cell_id=False, use_pert_idose=False):
        super(CIGER, self).__init__()
        self.fp_type = fp_type
        self.use_pert_type = use_pert_type
        self.use_cell_id = use_cell_id
        self.use_pert_idose = use_pert_idose
        if self.fp_type == 'neural':
            self.input_dim = 128 + 128
            self.drug_fp = NeuralFingerprint(drug_input_dim['atom'], drug_input_dim['bond'], conv_layer_sizes=[64, 64],
                                             output_size=128, degree_list=[0, 1, 2, 3, 4, 5], device=device)
        else:
            self.input_dim = drug_input_dim + 128
        if self.use_pert_type:
            self.input_dim += pert_type_input_dim
        if self.use_cell_id:
            self.input_dim += cell_id_input_dim
        if self.use_pert_idose:
            self.input_dim += pert_idose_input_dim
        self.encode_dim = encode_dim
        self.gene_embed = nn.Sequential(nn.Embedding(978, gene_input_dim).from_pretrained(gene_embed, freeze=True),
                                        nn.Linear(gene_input_dim, 128))
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.encode_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(2 * self.encode_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
        self.attention = Attention(self.encode_dim, n_layers=1, n_heads=1, pf_dim=self.encode_dim, dropout=0.1,
                                   device=device)
        self.initializer = initializer
        # self.init_weights()
        self.sigmoid = nn.Sigmoid()
        self.loss_type = loss_type
        self.label_type = label_type
        self.device = device

    def init_weights(self):
        if self.initializer is None:
            return
        for name, parameter in self.named_parameters():
            if parameter.dim() == 1:
                nn.init.constant_(parameter, 0.)
            else:
                self.initializer(parameter)

    def forward(self, input_drug, input_gene, input_pert_type, input_cell_id, input_pert_idose):
        # input_drug = [batch * drug_input_dim]
        # input_gene = [batch * num_gene]
        # input_pert_type = [batch * pert_type_input_dim]
        # input_cell_id = [batch * cell_id_input_dim]
        # input_pert_idose = [batch * pert_idose_input_dim]
        n_batch, num_gene = input_gene.shape
        if self.fp_type == 'neural':
            input_drug = self.drug_fp(input_drug)
        if self.use_pert_type:
            input_drug = torch.cat((input_drug, input_pert_type), dim=1)
        if self.use_cell_id:
            input_drug = torch.cat((input_drug, input_cell_id), dim=1)
        if self.use_pert_idose:
            input_drug = torch.cat((input_drug, input_pert_idose), dim=1)
        # input_drug = [batch * (drug_input_dim + pert_type_input_dim + cell_id_input_dim + pert_idose_input_dim)]
        input_drug = input_drug.unsqueeze(1)
        # input_drug = [batch * 1 * (drug_input_dim + pert_type_input_dim + cell_id_input_dim + pert_idose_input_dim)]
        input_drug = input_drug.repeat(1, num_gene, 1)
        # input_drug = [batch * num_gene * (drug_input_dim + pert_type_input_dim +
        # cell_id_input_dim + pert_idose_input_dim)]
        input_gene = self.gene_embed(input_gene)
        # input_gene = [batch * num_gene * gene_input_dim]
        input = torch.cat((input_drug, input_gene), dim=2)
        # input = [batch * num_gene * (drug_input_dim + gene_input_dim + pert_type_input_dim +
        # cell_id_input_dim + pert_idose_input_dim)]
        input_encode = self.encoder(input)
        # input = [batch * num_gene * encode_dim]
        input_attn, attn = self.attention(input_encode, None)
        # input_attn = [batch * num_gene * encode_dim]
        input_attn = torch.cat([input_encode, input_attn], dim=-1)
        # input_attn = [batch * num_gene * (2 * encode_dim)]
        output = self.decoder(input_attn)
        # output = [batch * num_gene * 1]
        if self.label_type == 'pos' or self.label_type == 'neg':
            out = self.sigmoid(output.squeeze(2))
        elif self.label_type == 'raw' or self.label_type == 'raw_reverse':
            out = output.squeeze(2)
        else:
            raise ValueError('Unknown label_type: %s' % self.label_type)
        # out = [batch * num_gene]
        return out

    def loss(self, label, predict):
        if self.loss_type == 'point_wise_mse':
            loss = point_wise_mse(label, predict)
        elif self.loss_type == 'pair_wise_ranknet':
            loss = pair_wise_ranknet(label, predict, self.device)
        elif self.loss_type == 'list_wise_listnet':
            loss = list_wise_listnet(label, predict)
        elif self.loss_type == 'list_wise_listmle':
            loss = list_wise_listmle(label, predict, self.device)
        elif self.loss_type == 'list_wise_rankcosine':
            loss = list_wise_rankcosine(label, predict)
        elif self.loss_type == 'list_wise_ndcg':
            loss = list_wise_ndcg(label, predict)
        else:
            raise ValueError('Unknown loss: %s' % self.loss_type)
        return loss
