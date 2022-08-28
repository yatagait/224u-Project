import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Seq_Classification(nn.Module):
    def __init__(self, args):
        super(Seq_Classification, self).__init__()
        self.args = args
        if 'base' in args.bert_model:
            self.hid_dim = 768
        elif 'xlarge' in args.bert_model:
            self.hid_dim = 2048
        elif 'large' in args.bert_model:
            self.hid_dim = 1024
        else:
            NotImplementedError

        self.relu = nn.ReLU()

        self.net_adaptive = nn.Sequential(nn.Linear(self.hid_dim, 64))

        self.bert_layer = AutoModel.from_pretrained(args.bert_model)

    def forward(self, x_input_ids, x_attn_masks, x_token_type_ids):
        last_state = self.bert_layer(x_input_ids, x_attn_masks, x_token_type_ids)['last_hidden_state'][:, 0]
        last_state = self.net_adaptive[0](last_state)

        final_feat = last_state
        return final_feat

    def mixup_data(self, xs, ys, xq, yq, lam):
        query_size = xq.shape[0]
        shuffled_index = torch.randperm(query_size)
        xs = xs[shuffled_index]
        ys = ys[shuffled_index]
        mixed_x = lam * xq + (1 - lam) * xs
        return mixed_x, yq, ys, lam

    def forward_metamix(self, x_input_ids, x_attn_masks, x_token_type_ids,
                        yq_new, lam):
        final_feat = self.bert_layer(x_input_ids, x_attn_masks, x_token_type_ids)['last_hidden_state'][:, 0]
        mixed_x, reweighted_query, reweighted_support, lam = self.mixup_data(final_feat, yq_new, final_feat, yq_new, lam)
        x = self.net_adaptive[0](mixed_x)
        return x, reweighted_query, reweighted_support, lam

    def functional_forward(self, x_input_ids, x_attn_masks, x_token_type_ids, weights):
        last_state = self.bert_layer(x_input_ids, x_attn_masks, x_token_type_ids)['last_hidden_state'][:, 0]
        last_state = F.linear(last_state, weights['0.weight'], weights['0.bias'])

        final_feat = last_state

        return final_feat
