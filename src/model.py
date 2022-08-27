# coding=utf-8

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parameter import Parameter
from utils.utils import SReLU_limited, cfg_Block, AverageMeter, DCN, gmm_criterion, gmm_sample
from utils.utils_nri import encode_onehot, gumbel_softmax, my_softmax, nll_gaussian, kl_categorical, kl_categorical_uniform
from src.modules_nri import MLPEncoder, CNNEncoder, MLPDecoder, RNNDecoder

class RAIN(nn.Module):
    def __init__(self, cfg_state_enc, cfg_ge_att, cfg_init, cfg_lstm, cfg_enc, cfg_dec, cfg_mu, cfg_sig, D_att, D_heads_num, block_type, att_type, act_type, dropout, sig=True, use_sample=True, pa=True, gt=False):
        super(RAIN, self).__init__()

        self.D_att = D_att
        self.heads_num = D_heads_num
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout
        self.nl = 'MS'
        self.act_type = act_type
        self.sig = sig
        if not self.sig:
            self.fixed_var = 5e-5
        self.use_sample = use_sample
        self.gt = gt
        self.pa = pa
        self.reg_norm = 1.
        self.final_bias = False
        
        # Raw data encoder
        self.state_enc = cfg_Block(block_type, cfg_state_enc, self.nl)

        # Graph Extraction transformer
        if not self.gt and self.pa:
            self.key_ct = cfg_Block(block_type, cfg_enc, self.nl, final_bias=self.final_bias)
            self.query_ct = cfg_Block(block_type, cfg_enc, self.nl, final_bias=self.final_bias)
            self.value_ct = cfg_Block(block_type, cfg_enc, self.nl, final_bias=self.final_bias)

        self.value = cfg_Block(block_type, cfg_enc, self.nl)  # used in decoder

        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_ge_att, self.nl)
        elif self.att_type == 'kqv':
            self.key = cfg_Block(block_type, cfg_enc, self.nl, final_bias=self.final_bias)
            self.query = cfg_Block(block_type, cfg_enc, self.nl, final_bias=self.final_bias)

        # Encoding / Decoding LSTM
        self.init_hidden = cfg_Block(block_type, cfg_init, self.nl)
        self.lstm = nn.GRU(*cfg_lstm)
        self.lstm_num = cfg_lstm[-1]

        self.dec = cfg_Block(block_type, cfg_dec, self.nl)
        self.mu_dec = cfg_Block(block_type, cfg_mu, self.nl)
        if self.sig:
            self.sig_dec = cfg_Block(block_type, cfg_sig, self.nl)

        if block_type == 'mlp':
            self.D_k = 1
            self.D_s = 1
        elif block_type == 'res':
            self.D_k = 1
            self.D_s = 1

        self.SR = None
        if self.act_type == 'srelu':
            self.SR = SReLU_limited()

        if self.act_type == 'nrelu':
            self.norm_param = Parameter(
                torch.tensor(1.0, dtype=torch.float, requires_grad=True)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def initialize(self, x):
        # x.shape = [batch_num, agent_num, lstm_dim]
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(batch_num * agent_num, -1)
        h = self.init_hidden(x)
        return h.reshape(self.lstm_num, batch_num * agent_num, -1)

    def encode(self, x, hidden):
        # x.shape = (len_enc - 1, batch_num, agent_num, lstm_dim) (after transposed)
        x = self.state_enc(x)
        x = x.transpose(1, 0)
        len_enc_m1 = x.shape[0]
        batch_num = x.shape[1]
        agent_num = x.shape[2]
        x = x.reshape(len_enc_m1, batch_num * agent_num, -1)
        output, hidden = self.lstm(x, hidden)
        return output, hidden, (len_enc_m1, batch_num, agent_num, x.shape[-1])

    def extract(self, output, shape, weight=None, final=False):
        if self.gt:
            return None, weight
        else:
            if final:
                return None, None
            else:
                len_enc_m1, batch_num, agent_num, lstm_dim = shape
                # start_time = time.time()
                if self.pa:
                    # output.shape = [len_enc_m1 = len_enc - 1, batch_num * agent_num, lstm_dim]
                    k_ct = self.key_ct(output)
                    q_ct = self.query_ct(output)
                    v_ct = self.value_ct(output)

                    # 1. Sequence contraction : merging time_series into weighted sum of agent vector with attention module
                    head_dim = lstm_dim // self.heads_num
                    assert head_dim * self.heads_num == lstm_dim, "embed_dim must be divisible by num_heads"

                    k_ct = self.key_ct(output).reshape(len_enc_m1, batch_num, agent_num, self.heads_num, head_dim)
                    q_ct = self.query_ct(output).reshape(len_enc_m1, batch_num, agent_num, self.heads_num, head_dim)
                    v_ct = self.value_ct(output).reshape(len_enc_m1, batch_num, agent_num, self.heads_num, head_dim)

                    # change order into (batch_num, self.heads_num, agent_num, len_enc_m1, lstm_dim)
                    k_ct = k_ct.permute(1, 3, 2, 0, 4)
                    q_ct = q_ct.permute(1, 3, 2, 0, 4)
                    v_ct = v_ct.permute(1, 3, 2, 0, 4)

                    k_cts = torch.stack([k_ct for _ in range(agent_num)], dim=-3).unsqueeze(-1)
                    q_cts = torch.stack([q_ct for _ in range(agent_num)], dim=-4).unsqueeze(-1)
                    v_cts = torch.stack([v_ct for _ in range(agent_num)], dim=-3)

                    attention_score = torch.softmax((torch.matmul(q_cts.transpose(-2, -1), k_cts) / math.sqrt(lstm_dim)).squeeze(-1), dim=-2)
                    output = torch.sum(attention_score * v_cts, dim=-2)  # sequence contracted in time dimension
                    output = output.permute(0, 2, 3, 1, 4).reshape(batch_num, agent_num, agent_num, -1)  # (batch_num, agent_num1, agent_num2, self.heads_num * lstm_dim)
                
                else:
                    assert self.heads_num == 1
                    output = output[-1].reshape(batch_num, agent_num, -1)
                    output = torch.stack([output for _ in range(agent_num)], dim=-2)
                    attention_score = None

                # print(f'extract time : {time.time()-start_time}')
                # 2. Graph extraction : exploit graph structure from merged agent vector with transformer
                
                weight = None
                if self.att_type == 'gat':
                    w = torch.cat((output, output.transpose(-3, -2)), dim=-1)  # swap two agent dimensions
                    if self.act_type == 'sigmoid':
                        mask = torch.eye(agent_num, agent_num).to(output.device)
                        mask = mask.float().masked_fill(mask == 1, float(-10000.)).masked_fill(mask == 0, float(0.0))
                        weight = torch.sigmoid(self.att(w).squeeze(-1) + mask)
                    elif self.act_type == 'tanh':
                        mask = torch.eye(agent_num, agent_num).to(output.device)
                        mask = 1 - (mask.float())
                        weight = torch.tanh(self.att(w).squeeze(-1)+0.5) * mask
                elif self.att_type == 'kqv':
                    # raise NotImplementedError
                    k_ge = self.key(output).unsqueeze(-1)
                    q_ge = self.query(output).unsqueeze(-1)
                    weight = torch.softmax((torch.matmul(q_ge.transpose(-2, -1), k_ge) / math.sqrt(lstm_dim)).squeeze(-1).squeeze(-1)+mask, dim=-2)
                    
                return attention_score, weight

    def decode(self, x, hidden, weight):
        epsilon = 1e-6
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = self.state_enc(x)
        x = x.reshape(1, batch_num * agent_num, -1)
        output, hidden = self.lstm(x, hidden)

        b = hidden[-1].reshape(batch_num, agent_num, -1)
        v = self.value(b)

        p_list = [v]
        if self.att_type == 'gat':
            p_list.append(torch.bmm(weight, v) / agent_num)
        else:
            p_list.append(torch.bmm(weight, v))

        c = torch.cat(p_list, dim=-1)
        d = self.dec(c)
        mu = self.mu_dec(d)
        if self.sig:
            sig = (torch.sigmoid(self.sig_dec(d)).squeeze() + epsilon) / self.reg_norm
            return (mu, sig), hidden
        else:
            return (mu, torch.ones_like(mu) * self.fixed_var), hidden

    def model_train_rec(self, train_loader, model, criterion, optimizer, epoch, sampler, local_rank, args):
        extend = True
        train_losses_list = []
        data_num = train_loader.dataset.tensors[0].shape[1]   # length of train_data_enc, 50
        step_num = train_loader.dataset.tensors[1].shape[1]   # length of train_data_dec, 50
        agent_num = train_loader.dataset.tensors[1].shape[2]  # number of agents, 5 / 30
        state_num = train_loader.dataset.tensors[1].shape[3]  # number of state, 3 (dphi, sinphi, omega)
        
        for i in range(step_num):
            train_losses_list.append(AverageMeter("Loss_" + str(i), ":.4e"))
        train_losses_list.append(AverageMeter("Total_Loss" + str(i), ":.4e"))
        forcing_period = args.forcing_period
        model.train()
        
        for i, (data_enc, data_dec, *relations) in enumerate(train_loader):
            data = data_enc.cuda(local_rank)
            labels = data_dec.cuda(local_rank)
            edges = relations[0].cuda(local_rank)
            hidden = model.module.initialize(data[:, 0])
            if extend:
                output, hidden, shape = model.module.encode(data[:, :-1], hidden)
                _, weight = model.module.extract(output, shape, weight=(edges / 2.0 if args.gt else None))  # use final layer's hidden state
                data = data[:, -1]
                data_tmp = torch.cat((data_enc[:, -1].unsqueeze(1), data_dec), dim=1)
            else:
                output, _, shape = model.module.encode(data[:, :-1], hidden)
                _, weight = model.module.extract(output, shape, weight=(edges / 2.0 if args.gt else None))  # use final layer's hidden state
                data = data[:, 0]
                data_tmp = torch.cat((data_enc, data_dec[:, 0].unsqueeze(1)), dim=1)

            label_diff_list = (data_tmp[:, 1:] - data_tmp[:, :-1]).cuda(local_rank)
            optimizer.zero_grad()

            for n in range(step_num):
                label = labels[:, n]
                label_diff = label_diff_list[:, n]

                (mu, sig), hidden = model.module.decode(data, hidden, weight)
                if args.use_sample:
                    sample = sampler(mu, sig).cuda(local_rank)
                else:
                    sample = mu.cuda(local_rank)

                # Teacher forcing (depends on epoch)

                if args.ww:
                    next_data = sample + data[:, :, :-1]
                else:
                    next_data = sample + data

                if args.teacher_forcing != "None":
                    if epoch < args.forcing_period or args.teacher_forcing == "tf3":
                        if args.teacher_forcing == "tf":
                            next_data_mask = (
                                torch.bernoulli(
                                    torch.ones((sample.shape[0], sample.shape[1], 1))
                                    * F.relu(torch.tensor(1 - (epoch + 1) / forcing_period))
                                ).cuda(local_rank)
                            )
                        
                        elif args.teacher_forcing == "tf2":
                            next_data_mask = (
                                torch.ones((sample.shape[0], sample.shape[1], 1)) * int(n / step_num > epoch / forcing_period)).cuda(local_rank)  # if the curretn step (ratio) exceeds current epoch (ratio), fill the maks to 1 for teacher forcing

                        else:  # tf3
                            interval = 10
                            if n % interval == interval - 1:
                                next_data_mask = (torch.ones((sample.shape[0], sample.shape[1], 1))).cuda(local_rank)
                            else:
                                next_data_mask = (torch.zeros((sample.shape[0], sample.shape[1], 1))).cuda(local_rank)
                        
                        if args.ww:
                            next_data = (next_data_mask * label[:, :, :-1] + (1 - next_data_mask) * next_data)  # W
                        else:
                            next_data = (next_data_mask * label + (1 - next_data_mask) * next_data)   # noW

                if args.ww:
                    data = torch.cat((next_data, data[:, :, -1].unsqueeze(-1)), dim=-1)  # W
                else:
                    data = next_data  # noW

                if args.diff:
                    if args.ww:
                        train_loss = criterion(label_diff[..., :-1], mu, sig).mean()
                    else:
                        train_loss = criterion(label_diff, mu, sig).mean()
                else:
                    if args.ww:
                        train_loss = criterion(label[..., :-1], mu, sig).mean()
                    else:
                        train_loss = criterion(label, mu, sig).mean()
                
                train_loss.backward(retain_graph=True)
                train_losses_list[n].update(train_loss.item(), data.size(0) * agent_num)
                train_losses_list[-1].update(train_loss.item(), data.size(0) * agent_num)

            optimizer.step()
  
            del mu, sig, hidden
            del train_loss, sample, next_data, data
            del data_tmp, label_diff, label_diff_list
            torch.cuda.empty_cache()

        return (
            [train_losses_list[i].avg for i in range(len(train_losses_list))],
            [train_losses_list[i].count for i in range(len(train_losses_list))],
        )

    def model_test(self, test_loader, model, criterion, sampler, local_rank, args):

        test_losses_list = []
        data_num = test_loader.dataset.tensors[0].shape[1]  # length of train_data_enc, 50
        step_num = test_loader.dataset.tensors[1].shape[1]  # length of train_data_dec, 50
        agent_num = test_loader.dataset.tensors[1].shape[2] # number of agents, 5 / 30
        state_num = test_loader.dataset.tensors[1].shape[3] # number of state, 3 (dphi, sinphi, omega)

        for i in range(step_num):
            test_losses_list.append(AverageMeter("Loss_" + str(i), ":.4e"))
        test_losses_list.append(AverageMeter("Total_Loss" + str(i), ":.4e"))
        model.eval()

        with torch.no_grad():
            for i, (data_enc, data_dec, *relations) in enumerate(test_loader):
                data = data_enc.cuda(local_rank)
                labels = data_dec.cuda(local_rank)
                edges = relations[0].cuda(local_rank) if relations else None
                hidden = model.module.initialize(data[:, 0])
                output, hidden, shape = model.module.encode(data[:, :-1], hidden)
                _, weight = model.module.extract(output, shape, weight=(edges / 2.0 if args.gt else None))  # use final layer's hidden state
                data = data[:, -1]
                pred_mu = []
                pred_sig = []
                data_tmp = torch.cat((data_enc[:, -1].unsqueeze(1), data_dec), dim=1)
                label_diff_list = (data_tmp[:, 1:] - data_tmp[:, :-1]).cuda(local_rank)

                for n in range(step_num):
                    (mu, sig), hidden = model.module.decode(data, hidden, weight)
                    
                    if args.use_sample:
                        sample = sampler(mu, sig).cuda(local_rank)
                    else:
                        sample = mu.cuda(local_rank)

                    if args.ww:
                        next_data = sample + data[:, :, :-1]
                        data = torch.cat((next_data, data[:, :, -1].unsqueeze(-1)), dim=-1)
                    else:
                        next_data = sample + data
                        data = next_data

                    if args.diff:
                        pred_mu.append(mu)
                    else:
                        pred_mu.append(next_data)
                    pred_sig.append(sig)

                pred_mu = torch.stack(pred_mu, dim=1)
                pred_sig = torch.stack(pred_sig, dim=1)

                if args.diff:
                    if args.ww:
                        test_loss = criterion(label_diff_list[..., :-1], pred_mu, pred_sig)
                    else:
                        test_loss = criterion(label_diff_list, pred_mu, pred_sig)
                else:
                    if args.ww:
                        test_loss = criterion(labels[..., :-1], pred_mu, pred_sig)
                    else:
                        test_loss = criterion(labels, pred_mu, pred_sig)

            test_loss = torch.mean(test_loss, dim=[0, 2, 3])  # contract excep time dimension

            for i in range(step_num):
                test_losses_list[i].update(test_loss[i].item(), data.size(0) * agent_num)

            test_loss = torch.mean(test_loss)
            test_losses_list[-1].update(test_loss.item(), data.size(0) * step_num * agent_num)

            del mu, sig, hidden, pred_mu, pred_sig
            del test_loss, sample, next_data, data
            del label_diff_list, data_tmp
            torch.cuda.empty_cache()

        test_index = np.random.randint(test_loader.dataset.tensors[0].shape[0])
        output = RAIN.trajectory_plot(test_loader, model, index=test_index, mean=True, predict=True, ww=args.ww, gt=args.gt)

        return (
            [test_losses_list[i].avg for i in range(len(test_losses_list))],
            [test_losses_list[i].count for i in range(len(test_losses_list))],
            output
        )

    def model_train(self, train_loader, model, criterion, optimizer, epoch, sampler, local_rank, args):
        extend = True
        train_losses_list = []
        data_num = train_loader.dataset.tensors[0].shape[1]   # length of train_data_enc, 50
        step_num = train_loader.dataset.tensors[1].shape[1]   # length of train_data_dec, 50
        agent_num = train_loader.dataset.tensors[1].shape[2]  # number of agents, 5 / 30
        state_num = train_loader.dataset.tensors[1].shape[3]  # number of state, 3 (dphi, sinphi, omega)
        
        for i in range(step_num):
            train_losses_list.append(AverageMeter("Loss_" + str(i), ":.4e"))
        train_losses_list.append(AverageMeter("Total_Loss" + str(i), ":.4e"))
        forcing_period = args.forcing_period
        model.train()
        
        for i, (data_enc, data_dec, *relations) in enumerate(train_loader):
            data = data_enc.cuda(local_rank)
            labels = data_dec.cuda(local_rank)
            edges = relations[0].cuda(local_rank) if relations else None
            hidden = model.module.initialize(data[:, 0])
            if extend:
                output, hidden, shape = model.module.encode(data[:, :-1], hidden)
                _, weight = model.module.extract(output, shape, weight=(edges / 2.0 if args.gt else None))  # use final layer's hidden state
                data = data[:, -1]
                data_tmp = torch.cat((data_enc[:, -1].unsqueeze(1), data_dec), dim=1)
            else:
                output, _, shape = model.module.encode(data[:, :-1], hidden)
                _, weight = model.module.extract(output, shape, weight=(edges / 2.0 if args.gt else None))  # use final layer's hidden state
                data = data[:, 0]
                data_tmp = torch.cat((data_enc, data_dec[:, 0].unsqueeze(1)), dim=1)

            label_diff_list = (data_tmp[:, 1:] - data_tmp[:, :-1]).cuda(local_rank)
            optimizer.zero_grad()
            pred_mu = []
            pred_sig = []

            for n in range(step_num):
                label = labels[:, n]
                label_diff = label_diff_list[:, n]

                (mu, sig), hidden = model.module.decode(data, hidden, weight)
                if args.use_sample:
                    sample = sampler(mu, sig).cuda(local_rank)
                else:
                    sample = mu.cuda(local_rank)

                # Teacher forcing (depends on epoch)

                if args.ww:
                    next_data = sample + data[:, :, :-1]
                else:
                    next_data = sample + data

                if args.teacher_forcing != "None":
                    if epoch < args.forcing_period or args.teacher_forcing == "tf3":
                        if args.teacher_forcing == "tf":
                            next_data_mask = (
                                torch.bernoulli(
                                    torch.ones((sample.shape[0], sample.shape[1], 1))
                                    * F.relu(torch.tensor(1 - (epoch + 1) / forcing_period))
                                ).cuda(local_rank)
                            )
                        
                        elif args.teacher_forcing == "tf2":
                            next_data_mask = (
                                torch.ones((sample.shape[0], sample.shape[1], 1)) * int(n / step_num > epoch / forcing_period)).cuda(local_rank)  # if the curretn step (ratio) exceeds current epoch (ratio), fill the maks to 1 for teacher forcing

                        else:  # tf3
                            interval = 10
                            if n % interval == interval - 1:
                                next_data_mask = (torch.ones((sample.shape[0], sample.shape[1], 1))).cuda(local_rank)
                            else:
                                next_data_mask = (torch.zeros((sample.shape[0], sample.shape[1], 1))).cuda(local_rank)
                        
                        if args.ww:
                            next_data = (next_data_mask * label[:, :, :-1] + (1 - next_data_mask) * next_data)  # W
                        else:
                            next_data = (next_data_mask * label + (1 - next_data_mask) * next_data)   # noW

                if args.ww:
                    data = torch.cat((next_data, data[:, :, -1].unsqueeze(-1)), dim=-1)  # W
                else:
                    data = next_data

                if args.diff:
                    pred_mu.append(mu)
                else:
                    pred_mu.append(next_data)
                pred_sig.append(sig)

            pred_mu = torch.stack(pred_mu, dim=1)
            pred_sig = torch.stack(pred_sig, dim=1)

            if args.diff:
                if args.ww:
                    train_loss = criterion(label_diff_list[..., :-1], pred_mu, pred_sig).mean()
                else:
                    train_loss = criterion(label_diff_list, pred_mu, pred_sig).mean()
            else:
                if args.ww:
                    train_loss = criterion(labels[..., :-1], pred_mu, pred_sig).mean()
                else:
                    train_loss = criterion(labels, pred_mu, pred_sig).mean()

            train_loss.backward()
            optimizer.step()

            for n in range(step_num):
                train_losses_list[n].update(-1, data.size(0) * agent_num)
                train_losses_list[-1].update(train_loss.item(), data.size(0) * agent_num * step_num)

            del mu, sig, hidden, pred_mu, pred_sig
            del train_loss, sample, next_data, data
            del data_tmp, label_diff, label_diff_list
            torch.cuda.empty_cache()

        return (
            [train_losses_list[i].avg for i in range(len(train_losses_list))],
            [train_losses_list[i].count for i in range(len(train_losses_list))],
        )

    @staticmethod
    def trajectory_plot(test_loader, model, index, mean=True, predict=True, ww=True, gt=False):
        i = index
        data_enc = test_loader.dataset.tensors[0][i].unsqueeze(0).cuda()
        data_dec = test_loader.dataset.tensors[1][i].unsqueeze(0).cuda()
        edge_TF = len(test_loader.dataset.tensors) > 2
        edges = test_loader.dataset.tensors[2][i].unsqueeze(0).cuda() if edge_TF else torch.ones(10)  ## placeholder
        
        model.eval()
        criterion = gmm_criterion(1)
        sampler = gmm_sample(1)
        data_num = test_loader.dataset.tensors[0].shape[1]
        step_num = test_loader.dataset.tensors[1].shape[1]
        
        hidden = model.module.initialize(data_enc[:, 0])
        output, hidden, shape = model.module.encode(data_enc[:, :-1], hidden)
        attention_score, weight = model.module.extract(output, shape, weight=(edges / 2.0 if gt else None))  # use final layer's hidden state
        data = data_enc[:, -1]
        
        if predict:
            prediction_list = []
            test_loss_list = torch.zeros(step_num)
            data_tmp = torch.cat((data_enc[:, -1].unsqueeze(1), data_dec), dim=1)
            
            for n in range(step_num):
                (mu, sig), hidden = model.module.decode(data, hidden, weight)
                #nll = criterion(label_diff[0][:, :2], mu, sig)
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
                    # test_loss_list[n] = (torch.mean(nll))

            return data_enc, data_dec, DCN(edges).squeeze(), attention_score, DCN(weight).squeeze(), (torch.cat(prediction_list, dim=0).transpose(1, 0)), test_loss_list
        else:
            return data_enc, data_dec, edges, attention_score, weight

class NRI(nn.Module):
    def __init__(self, encoder_type, decoder_type, state_num, D_hidden_enc, D_hidden_dec, edge_types, agent_num, input_length, ww, factor=True, dropout=0.0, skip_first=False, extend=False):
        super(NRI, self).__init__()
        self.act_type = 'RL'
        self.encoder = None
        self.decoder = None
        self.state_num = state_num
        self.edge_types = edge_types
        self.factor = factor
        self.dropout = dropout
        self.skip_first = skip_first
        self.extend = extend
        self.ww = ww

        self.off_diag = np.ones([agent_num, agent_num]) - np.eye(agent_num)
        self.rel_rec = torch.FloatTensor(np.array(encode_onehot(np.where(self.off_diag)[0]), dtype=np.float32)).cuda()
        self.rel_send = torch.FloatTensor(np.array(encode_onehot(np.where(self.off_diag)[1]), dtype=np.float32)).cuda()

        if encoder_type == 'mlp':
            self.encoder = MLPEncoder(input_length * self.state_num, D_hidden_enc, self.edge_types, self.dropout, self.factor)
        elif encoder_type == 'cnn':
            self.encoder = CNNEncoder(self.state_num, self.D_hidden, self.edge_types, self.dropout, self.factor)

        if decoder_type == 'mlp':
            self.decoder = MLPDecoder(n_in_node=self.state_num,
                                edge_types=self.edge_types,
                                msg_hid=D_hidden_dec,
                                msg_out=D_hidden_dec,
                                n_hid=D_hidden_dec,
                                do_prob=self.dropout,
                                skip_first=self.skip_first,
                                ww=self.ww)
        elif decoder_type == 'rnn':
            self.decoder = RNNDecoder(n_in_node=self.state_num,
                                edge_types=self.edge_types,
                                n_hid=D_hidden_dec,
                                do_prob=self.dropout,
                                skip_first=self.skip_first,
                                ww=self.ww)

    def model_train(self, train_loader, model, criterion, optimizer, epoch, sampler, local_rank, args, temp=0.5, hard=False, prior=False):

        nll_train = []
        log_prior = [0.91, 0.03, 0.03, 0.03]

        model.train()
        prediction_steps = 10

        for batch_idx, (data_enc, data_dec, *relations) in enumerate(train_loader):
            data_enc = data_enc.cuda(local_rank).transpose(2, 1)
            data_dec = data_dec.cuda(local_rank).transpose(2, 1)
            if self.extend:
                data_pred = torch.cat((data_enc, data_dec), dim=2)
            else:
                data_pred = torch.cat((data_enc, data_dec[:, :, 0].unsqueeze(2)), dim=2)
            optimizer.zero_grad()
            logits = model.module.encoder(data_enc, self.rel_rec, self.rel_send)
            edges = gumbel_softmax(logits, tau=temp, hard=hard)
            prob = my_softmax(logits, -1)

            if self.extend:
                output = model.module.decoder(data_pred, edges, self.rel_rec, self.rel_send, prediction_steps, burn_in=True, burn_in_steps=args.input_length - 1)
                target = data_pred[:, :, args.input_length:, :]
                output = output[:, :, -args.output_length:, :]
            else:
                output = model.module.decoder(data_pred, edges, self.rel_rec, self.rel_send, prediction_steps, burn_in=False)
                target = data_pred[:, :, 1:, :]

            if args.ww:
                output = output[..., :-1]
                target = target[..., :-1]

            loss_nll = nll_gaussian(output, target, 5e-5)
            if prior:
                loss_kl = kl_categorical(prob, log_prior, args.agent_num)
            else:
                loss_kl = kl_categorical_uniform(prob, args.agent_num, self.edge_types)
            loss = loss_nll + loss_kl

            loss.backward()
            optimizer.step()
            nll_train.append(loss_nll.item())
        return [np.mean(nll_train)], None

    def model_test(self, test_loader, model, criterion, sampler, local_rank, args, temp=0.5, hard=False):

        nll_test = []
        model.eval()
        prediction_steps = 100

        for batch_idx, (data_enc, data_dec, *relations) in enumerate(test_loader):
            data_enc = data_enc.cuda(local_rank).transpose(2, 1)
            data_dec = data_dec.cuda(local_rank).transpose(2, 1)
            data_pred = torch.cat((data_enc, data_dec), dim=2)

            logits = model.module.encoder(data_enc, self.rel_rec, self.rel_send)
            edges = gumbel_softmax(logits, tau=temp, hard=hard)

            output = model.module.decoder(data_pred, edges, self.rel_rec, self.rel_send, prediction_steps, burn_in=True, burn_in_steps=args.input_length - 1)
            target = data_pred[:, :, args.input_length:, :]
            output = output[:, :, -args.output_length:, :]

            loss_nll = nll_gaussian(output, target, 5e-5)
            nll_test.append(loss_nll.item())

        test_index = np.random.randint(test_loader.dataset.tensors[0].shape[0])
        output = NRI.trajectory_plot(test_loader, model, test_index, args)
        
        return [np.mean(nll_test)], None, output

    def trajectory_plot(test_loader, model, index, args, predict=True, temp=0.5, hard=False):

        i = index
        data_enc = test_loader.dataset.tensors[0][i].unsqueeze(0).cuda()
        data_dec = test_loader.dataset.tensors[1][i].unsqueeze(0).cuda()
        edge_TF = len(test_loader.dataset.tensors) > 2
        edges_label = test_loader.dataset.tensors[2][i].unsqueeze(0).cuda() if edge_TF else torch.ones(10)  ## placeholder

        off_diag = np.ones([args.agent_num, args.agent_num]) - np.eye(args.agent_num)
        rel_rec = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)).cuda()
        rel_send = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)).cuda()

        model.eval()
        prediction_steps = 100
        if predict:
            data_enc = data_enc.transpose(2, 1)
            data_dec = data_dec.transpose(2, 1)
            data_pred = torch.cat((data_enc, data_dec), dim=2)

            logits = model.module.encoder(data_enc, rel_rec, rel_send)
            edges = gumbel_softmax(logits, tau=temp, hard=hard)
            _, edges_pred = logits.max(-1)

            output = model.module.decoder(data_pred, edges, rel_rec, rel_send, prediction_steps, burn_in=True, burn_in_steps=args.input_length - 1)
            output = output[:, :, -args.output_length:, :]

            edges_label = DCN(edges_label).squeeze()
            edges_pred = DCN(edges_pred).squeeze()

            edges_pred = np.insert(edges_pred, np.arange(args.agent_num) * args.agent_num, 0).reshape(args.agent_num, args.agent_num)

            return data_enc.transpose(2, 1), data_dec.transpose(2, 1), edges_label, None, edges_pred, output[0], None
        else:
            return data_enc, data_dec, edges, None, edges_pred

class JointLSTM(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_atoms, n_layers, do_prob=0., ww=False):
        super(JointLSTM, self).__init__()
        self.act_type = 'RL'
        self.n_atoms = n_atoms
        self.n_in = n_in
        self.n_out = n_out
        self.ww = ww
        self.var = 5e-5

        self.fc1_1 = nn.Linear(n_in, n_hid)
        self.fc1_2 = nn.Linear(n_hid, n_hid)
        self.rnn = nn.LSTM(n_atoms * n_hid, n_atoms * n_hid, n_layers)
        self.fc2_1 = nn.Linear(n_atoms * n_hid, n_atoms * n_hid)
        self.fc2_2 = nn.Linear(n_atoms * n_hid, n_atoms * n_out)

        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def step(self, ins, hidden=None):
        # Input shape: [num_sims, n_atoms, n_in]
        x = F.relu(self.fc1_1(ins))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.relu(self.fc1_2(x))
        x = x.view(ins.size(0), -1)
        # [num_sims, n_atoms*n_hid]

        x = x.unsqueeze(0)
        x, hidden = self.rnn(x, hidden)
        x = x[0, :, :]

        x = F.relu(self.fc2_1(x))
        x = self.fc2_2(x)
        # [num_sims, n_out*n_atoms]

        x = x.view(ins.size(0), ins.size(1), -1)
        # [num_sims, n_atoms, n_out]

        # Predict position/velocity difference
        if self.ww:
            x = x + ins[..., :-1]
            x = torch.cat((x, ins[..., -1].unsqueeze(-1)), dim=-1)
        else:
            x = x + ins

        return x, hidden

    def forward(self, inputs, prediction_steps, burn_in=False, burn_in_steps=1):

        # Input shape: [num_sims, num_things, num_timesteps, n_in]

        outputs = []
        hidden = None

        for step in range(0, inputs.size(2) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, :, step, :]
                else:
                    ins = outputs[step - 1]
            else:
                # Use ground truth trajectory input vs. last prediction
                if not step % prediction_steps:
                    ins = inputs[:, :, step, :]
                else:
                    ins = outputs[step - 1]

            output, hidden = self.step(ins, hidden)

            # Predict position/velocity difference
            outputs.append(output)

        outputs = torch.stack(outputs, dim=2)

        return outputs

    def model_train(self, train_loader, model, criterion, optimizer, epoch, sampler, local_rank, args):

        nll_train = []
        model.train()
        
        for i, (data_enc, data_dec, *relations) in enumerate(train_loader):
            data_enc = data_enc.cuda(local_rank).transpose(2, 1)
            data_dec = data_dec.cuda(local_rank).transpose(2, 1)
            data_pred = torch.cat((data_enc, data_dec), dim=2)
            optimizer.zero_grad()

            output = model(data_pred, args.input_length + args.output_length + 1,
                        burn_in=True,
                        burn_in_steps=args.input_length)

            target = data_pred[:, :, 1:, :]

            if args.ww:
                output = output[..., :-1]
                target = target[..., :-1]

            loss = nll_gaussian(output, target, self.var)

            loss.backward()
            optimizer.step()

            nll_train.append(loss.item())
            torch.cuda.empty_cache()

        return [np.mean(nll_train)], None

    def model_test(self, test_loader, model, criterion, sampler, local_rank, args):

        nll_test = []
        model.eval()
        
        for i, (data_enc, data_dec, *relations) in enumerate(test_loader):
            data_enc = data_enc.cuda(local_rank).transpose(2, 1)
            data_dec = data_dec.cuda(local_rank).transpose(2, 1)
            data_pred = torch.cat((data_enc, data_dec), dim=2)

            output = model(data_pred, args.input_length + args.output_length + 1,
                        burn_in=True,
                        burn_in_steps=args.input_length)

            target = data_pred[:, :, 1:, :]

            if args.ww:
                output = output[..., :-1]
                target = target[..., :-1]

            loss = nll_gaussian(output, target, self.var)
            nll_test.append(loss.item())
            torch.cuda.empty_cache()

        test_index = np.random.randint(test_loader.dataset.tensors[0].shape[0])
        output = JointLSTM.trajectory_plot(test_loader, model, test_index, args)

        return [np.mean(nll_test)], None, output

    @staticmethod
    def trajectory_plot(test_loader, model, index, args, predict=True):
        
        i = index
        data_enc = test_loader.dataset.tensors[0][i].unsqueeze(0).cuda()
        data_dec = test_loader.dataset.tensors[1][i].unsqueeze(0).cuda()
        edge_TF = len(test_loader.dataset.tensors) > 2
        edges_label = test_loader.dataset.tensors[2][i].unsqueeze(0).cuda() if edge_TF else torch.ones(10)  ## placeholder

        model.eval()

        if predict:
            data_enc = data_enc.transpose(2, 1)
            data_dec = data_dec.transpose(2, 1)
            data_pred = torch.cat((data_enc, data_dec), dim=2)

            output = model(data_pred, args.input_length + args.output_length + 1,
                        burn_in=True,
                        burn_in_steps=args.input_length)

            output = output[:, :, -args.output_length:, :]

            return data_enc.transpose(2, 1), data_dec.transpose(2, 1), edges_label, None, torch.ones_like(edges_label), output[0], None
        else:
            return data_enc, data_dec, edges_label, None, torch.ones_like(edges_label)

class SingleLSTM(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_atoms, n_layers, do_prob=0., ww=False):
        super(SingleLSTM, self).__init__()
        self.act_type = 'RL'
        self.n_atoms = n_atoms
        self.n_in = n_in
        self.n_out = n_out
        self.ww = ww
        self.var = 5e-5

        self.fc1_1 = nn.Linear(n_in, n_hid)
        self.fc1_2 = nn.Linear(n_hid, n_hid)
        self.rnn = nn.LSTM(n_hid, n_hid, n_layers)
        self.fc2_1 = nn.Linear(n_hid, n_hid)
        self.fc2_2 = nn.Linear(n_hid, n_out)

        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def step(self, ins, hidden=None):
        # Input shape: [num_sims, n_atoms, n_in]
        x = F.relu(self.fc1_1(ins))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.relu(self.fc1_2(x))

        x, hidden = self.rnn(x, hidden)
 
        x = F.relu(self.fc2_1(x))
        x = self.fc2_2(x)
        # [num_sims, n_out*n_atoms]

        # Predict position/velocity difference
        if self.ww:
            x = x + ins[..., :-1]
            x = torch.cat((x, ins[..., -1].unsqueeze(-1)), dim=-1)
        else:
            x = x + ins

        return x, hidden

    def forward(self, inputs, prediction_steps, burn_in=False, burn_in_steps=1, return_hidden=False):

        # Input shape: [num_sims, num_things, num_timesteps, n_in]

        outputs = []
        hidden = None

        for step in range(0, inputs.size(2) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, :, step, :]
                else:
                    ins = outputs[step - 1]
            else:
                # Use ground truth trajectory input vs. last prediction
                if not step % prediction_steps:
                    ins = inputs[:, :, step, :]
                else:
                    ins = outputs[step - 1]

            output, hidden = self.step(ins, hidden)

            # Predict position/velocity difference
            outputs.append(output)

        outputs = torch.stack(outputs, dim=2)

        if return_hidden:
            return outputs, hidden 
        else:
            return outputs

    def model_train(self, train_loader, model, criterion, optimizer, epoch, sampler, local_rank, args):

        nll_train = []
        model.train()
        
        for i, (data_enc, data_dec, *relations) in enumerate(train_loader):
            data_enc = data_enc.cuda(local_rank).transpose(2, 1)
            data_dec = data_dec.cuda(local_rank).transpose(2, 1)
            data_pred = torch.cat((data_enc, data_dec), dim=2)
            optimizer.zero_grad()

            output = model(data_pred, args.input_length + args.output_length + 1,
                        burn_in=True,
                        burn_in_steps=args.input_length)

            target = data_pred[:, :, 1:, :]

            if args.ww:
                output = output[..., :-1]
                target = target[..., :-1]

            loss = nll_gaussian(output, target, self.var)

            loss.backward()
            optimizer.step()

            nll_train.append(loss.item())
            torch.cuda.empty_cache()

        return [np.mean(nll_train)], None

    def model_test(self, test_loader, model, criterion, sampler, local_rank, args):

        nll_test = []
        model.eval()
        
        for i, (data_enc, data_dec, *relations) in enumerate(test_loader):
            data_enc = data_enc.cuda(local_rank).transpose(2, 1)
            data_dec = data_dec.cuda(local_rank).transpose(2, 1)
            data_pred = torch.cat((data_enc, data_dec), dim=2)

            output = model(data_pred, args.input_length + args.output_length + 1,
                        burn_in=True,
                        burn_in_steps=args.input_length)

            target = data_pred[:, :, 1:, :]

            if args.ww:
                output = output[..., :-1]
                target = target[..., :-1]

            loss = nll_gaussian(output, target, self.var)
            nll_test.append(loss.item())
            torch.cuda.empty_cache()

        test_index = np.random.randint(test_loader.dataset.tensors[0].shape[0])
        output = JointLSTM.trajectory_plot(test_loader, model, test_index, args)

        return [np.mean(nll_test)], None, output

    @staticmethod
    def trajectory_plot(test_loader, model, index, args, predict=True):
        
        i = index
        data_enc = test_loader.dataset.tensors[0][i].unsqueeze(0).cuda()
        data_dec = test_loader.dataset.tensors[1][i].unsqueeze(0).cuda()
        edge_TF = len(test_loader.dataset.tensors) > 2
        edges_label = test_loader.dataset.tensors[2][i].unsqueeze(0).cuda() if edge_TF else torch.ones(10)  ## placeholder

        model.eval()

        if predict:
            data_enc = data_enc.transpose(2, 1)
            data_dec = data_dec.transpose(2, 1)
            data_pred = torch.cat((data_enc, data_dec), dim=2)

            output = model(data_pred, args.input_length + args.output_length + 1,
                        burn_in=True,
                        burn_in_steps=args.input_length)

            output = output[:, :, -args.output_length:, :]

            return data_enc.transpose(2, 1), data_dec.transpose(2, 1), edges_label, None, torch.ones_like(edges_label), output[0], None
        else:
            return data_enc, data_dec, edges_label, None, torch.ones_like(edges_label)