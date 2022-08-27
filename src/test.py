# coding=utf-8
import torch
from utils.utils import gmm_criterion, gmm_sample


def trajectory_plot(model, test_loader, index, mean=True, predict=True, ww=True, gt=False):
    i = index
    data_enc = test_loader.dataset.tensors[0][i].unsqueeze(0).cuda()
    data_dec = test_loader.dataset.tensors[1][i].unsqueeze(0).cuda()
    edges = test_loader.dataset.tensors[2][i].unsqueeze(0).cuda()
    freq = test_loader.dataset.tensors[3][i]
    order = test_loader.dataset.tensors[4][i]
    
    model.eval()
    criterion = gmm_criterion(1)
    sampler = gmm_sample(1)
    data_num = test_loader.dataset.tensors[0].shape[1]
    step_num = test_loader.dataset.tensors[1].shape[1]
    
    hidden, cell = model.module.initialize(data_enc[:, 0])
    output, (hidden, cell), shape = model.module.encode(data_enc[:, :-1], hidden, cell)
    attention_score, weight = model.module.extract(output, shape, weight=(edges / 2.0 if gt else None))  # use final layer's hidden state
    data = data_enc[:, -1]
    # print(weight.shape, edges.shape)
    
    if predict:
        prediction_list = []
        test_loss_list = torch.zeros(step_num)
        data_tmp = torch.cat((data_enc[:, -1].unsqueeze(1), data_dec), dim=1)
        label_diff_orig = (data_tmp[:, 1:] - data_tmp[:, :-1]).cuda()
        
        for n in range(step_num):  # labels.shape[1]
            label_diff = label_diff_orig[:, n]
            (mu, sig), hidden, cell = model.module.decode(data, hidden, cell, weight)
            nll = criterion(label_diff[0][:, :2], mu, sig)
            if predict:
                if mean:
                    sample = mu
                else:
                    sample = sampler(mu, sig)
            
                if ww:
                    next_data = sample.cuda() + data[:, :, :-1]
                    data = torch.cat((next_data, data[:, :, -1].unsqueeze(-1)), dim=-1)
                else:
                    next_data = sample.cuda() + data
                    data = next_data
                prediction_list.append(data)
                test_loss_list[n] = (torch.mean(nll))

        return data_enc, data_dec, edges, attention_score, weight, (torch.cat(prediction_list, dim=0).transpose(1, 0)), test_loss_list
    else:
        return data_enc, data_dec, edges, attention_score, weight
