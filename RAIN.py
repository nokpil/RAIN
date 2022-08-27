# coding=utf-8
import argparse
import os
import time
import numpy as np
import tracemalloc
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLRP
from torch.nn.parallel import DistributedDataParallel, DataParallel


from utils.utils import str2bool, dataloader_kuramoto, dataloader_spring, dataloader_charge, dataloader_motion, gmm_criterion, gmm_sample, group_weight, count_vars, DCN
from utils.logger import setup_logger_kwargs, EpochLogger
from src.model import RAIN, NRI, JointLSTM, SingleLSTM
from src.system import System

parser = argparse.ArgumentParser(description="Pytorch RAIN Training")

# Fixed
parser.add_argument("-d", "--distributed", default=True, type=str2bool, help="Distributed training when enabled ",)
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 0)")
parser.add_argument("--epochs", default=10000, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)",)
parser.add_argument("--lr", "--learning-rate", default=5e-4, type=float, metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--wd", "--weight-decay", default=0, type=float, metavar="W", help="weight decay (default: 0.000)", dest="weight_decay")
parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--dropout", default=0.0, type=float, help="Rate of dropout on attention.")
parser.add_argument("--forcing-period", default=30, type=int, help="Teacher forcing period")
parser.add_argument("--block-type", default="mlp", type=str, help="mlp : simple multi-layer perceptron, res : skip-connection")
parser.add_argument("--seed", default=42, type=int, help="Random seed for torch and numpy")

# Variables
parser.add_argument("--system", default="kuramoto", type=str, help="system : kuramoto, spring")
parser.add_argument("--model-type", default="RAIN", type=str, help="model type : lstm, RAIN, NRI")
parser.add_argument("--att-type", default="gat", type=str, help="kqv : normal kqv (linear transformation), gat")
parser.add_argument("--agent-num", default=10, type=int, help="Number of LSTM layers for encoding")
parser.add_argument("--dt", default=10, type=int, help="Number of LSTM layers for encoding")
parser.add_argument("--heads-dim", default=128, type=int, help="Dimension of a single head of attention vector.")
parser.add_argument("--heads-num", default=1, type=int, help='For "multi", works as number of heads.')
parser.add_argument("--mode-num", default=1, type=int, help="Number of gaussian mixture mode.")
parser.add_argument("--lstm-num", default=1, type=int, help="Number of LSTM layers for encoding")
parser.add_argument("-b", "--batch-size", default=16, type=int, metavar="N", help="mini-batch size (default: 32)")
parser.add_argument("--input-length", default=50, type=int, help="Input length of the sequence (max : 49)") 
parser.add_argument("--output-length", default=50, type=int, help="Input length of the sequence (max : 50)")
parser.add_argument("--noise-var", default=0., type=float, help="Noise strength.")
parser.add_argument("--act-type", default='sigmoid', type=str, help="Activation type.")
parser.add_argument('-it', '--interaction-type', type=str, default='N', help='interaction type / N (normal), S (signed), D (directed), SD (signed+directed)')
parser.add_argument('-sm', '--sample-mode', type=str, default='uniform', help='interaction weight type / uniform, normal, duplex')
parser.add_argument("--sig", default=True, type=str2bool, help="Whether using generated variance or not")
parser.add_argument("--use-sample", default=True, type=str2bool, help="Whether use generated mu or not")
parser.add_argument("--pa", default=True, type=str2bool, help="Whether use pairwise attention or not")
parser.add_argument("--gt", default=False, type=str2bool, help="Whether using GroundTruth weight")
parser.add_argument("--ww", default=False, type=str2bool, help="Whether using inherent frequency as a feature")
parser.add_argument("--diff", default=True, type=str2bool, help="data diff or raw")
parser.add_argument("--checkpoint", "-cp", default='-1', type=str, help="(Epoch of saved checkpoint of the model)_(New run's name). if not, -1.")
parser.add_argument('-tf', "--teacher-forcing", default="None", type=str, help="None : No teacher forcing, tf : Bernouli mask, tf2 : Linear mask")
parser.add_argument("--indicator", default="", type=str, help="Additional specification for file name.")


def main():
    tracemalloc.start()
    best_test_loss = np.inf
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(torch.cuda.device_count(), local_rank)
    if local_rank == 0:
        print('REC')
        print(f'sig : {args.sig}')
        print(f'use_sample : {args.use_sample}')
        print(f'pa : {args.pa}')
        print(f'gt : {args.gt}')
        print(f'ww : {args.ww}')
        print(f'diff : {args.diff}')
        if args.system == 'spring':
            print('data : ' + 'spring' + '_' + str(args.agent_num) + '_t' + str(int(args.dt)))
        elif args.system == 'kuramoto':
            print('data : ' + 'kuramoto' + '_' + str(args.agent_num) + '_t' + str(int(args.dt)) + '_' + args.sample_mode + '_' + args.interaction_type)
        elif args.system == 'charge':
            print('data : ' + 'charge' + '_' + str(args.agent_num) + '_t' + str(int(args.dt)))
        elif args.system == 'motion':
            print('data : ' + 'motion')
        else:
            raise NotImplementedError
    torch.distributed.init_process_group(backend="gloo", init_method="env://")
    torch.cuda.set_device(local_rank)
    n_GPUs = 1  # Can be modified according to local gpu settings
    test_freq = 10
    save_freq = 50
    color_list = ['#dd0066', '#ff7400', '#ffce00', '#0b9822', '#9966ff']  # for trajectory drawing

    # System parameters
    param_dict = {}
    param_dict["agent_num"] = args.agent_num
    param_dict["dt"] = args.dt
    param_dict["data_step"] = args.input_length
    param_dict["label_step"] = args.output_length
    if args.system == 'spring':
        param_dict["state_num"] = 4
        param_dict["answer_num"] = 4
        param_dict["const_num"] = 0
        system = System(name='Spring', param_dict=param_dict)
        file_name = f"{system.rule_name}_A{system.agent_num}_dt{args.dt}"
        indicator = f"AT{args.att_type}_BT{args.block_type}_HN{args.heads_num}_Dk{args.mode_num}_DL{args.lstm_num}_NV{args.noise_var}_AT{args.act_type}_{args.indicator}"

    elif args.system == 'kuramoto':
        if args.ww:
            param_dict["state_num"] = 3
            param_dict["answer_num"] = 2
            param_dict["const_num"] = 1
        else:
            param_dict["state_num"] = 2
            param_dict["answer_num"] = 2
            param_dict["const_num"] = 0
        system = System(name='Kuramoto', param_dict=param_dict)
        file_name = f"{system.rule_name}_A{system.agent_num}_dt{args.dt}"
        indicator = f"AT{args.att_type}_BT{args.block_type}_HN{args.heads_num}_Dk{args.mode_num}_DL{args.lstm_num}_NV{args.noise_var}_AT{args.act_type}_IT{str(args.interaction_type).lower()}_SM{args.sample_mode}_{args.indicator}"
    
    elif args.system == 'charge':
        param_dict["state_num"] = 4
        param_dict["answer_num"] = 4
        param_dict["const_num"] = 0
        system = System(name='Charge', param_dict=param_dict)
        file_name = f"{system.rule_name}_A{system.agent_num}_dt{args.dt}"
        indicator = f"AT{args.att_type}_BT{args.block_type}_HN{args.heads_num}_Dk{args.mode_num}_DL{args.lstm_num}_NV{args.noise_var}_AT{args.act_type}_{args.indicator}"

    elif args.system == 'motion':
        param_dict["state_num"] = 6
        param_dict["answer_num"] = 6
        param_dict["const_num"] = 0
        system = System(name='Motion', param_dict=param_dict)
        file_name = f"{system.rule_name}"
        indicator = f"AT{args.att_type}_BT{args.block_type}_HN{args.heads_num}_Dk{args.mode_num}_DL{args.lstm_num}_NV{args.noise_var}_AT{args.act_type}_{args.indicator}"

    else:
        raise NotImplementedError

    # Data loading code

    exp_name = file_name + '_' + indicator
    checkpoint = (args.checkpoint).split('_')
    checkpoint_epoch, checkpoint_lr, checkpoint_name = '-1', '', ''
    if len(checkpoint) > 1:
        args.teacher_forcing = None
        checkpoint_epoch, checkpoint_name = checkpoint
        checkpoint_epoch = checkpoint_epoch.split('*')
        if len(checkpoint_epoch) > 1:
            checkpoint_epoch, checkpoint_lr = checkpoint_epoch
        else:
            checkpoint_epoch = checkpoint_epoch[0]
    exp_name_cp = ''
    if int(checkpoint_epoch) >= 0:
        exp_name_cp = exp_name
        exp_name += '_cp_' + checkpoint_name

    if args.teacher_forcing is None or args.model_type != 'RAIN':
        args.forcing_period = -1

    rank_0 = False
    if local_rank == 0:
        print(exp_name)
        rank_0 = True
        logger_kwargs = setup_logger_kwargs(exp_name, args.seed)
        print('Logger initiated')
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())
    

    if args.model_type == "RAIN":

        # Dimension definition
        D_head = args.heads_dim
        D_heads_num = args.heads_num
        D_att = D_head * D_heads_num

        D_hidden_lstm = 256
        D_in_lstm = system.state_num
        D_lstm_num = args.lstm_num

        D_in_enc = D_hidden_lstm
        D_hidden_enc = 256
        D_out_enc = D_att

        D_in_dec = D_out_enc * 2
        D_hidden_dec = 256
        D_out_dec = 256
        D_hidden_stat = 256

        # cfg definition
        cfg_state_enc = [D_in_lstm, 64, D_att]
        if args.pa:
            cfg_ge_att = [D_att * 2, 32, 16, 1]  # 32?
        else:
            cfg_ge_att = [D_hidden_lstm * 2, 32, 16, 1]
        
        cfg_init = [D_in_lstm, D_hidden_lstm * D_lstm_num]
        cfg_lstm = [D_att, D_hidden_lstm, D_lstm_num]

        cfg_enc = [D_in_enc, D_out_enc]
        cfg_dec = [D_in_dec, D_hidden_dec, D_out_dec]

        cfg_mu = [D_out_dec, D_hidden_stat, system.state_num - system.const_num]
        cfg_sig = [D_out_dec, D_hidden_stat, system.state_num - system.const_num]

        # model definition

        model = RAIN(
            cfg_state_enc,
            cfg_ge_att,
            cfg_init,
            cfg_lstm,
            cfg_enc,
            cfg_dec,
            cfg_mu,
            cfg_sig,
            D_att,
            D_heads_num,
            args.block_type,
            args.att_type,
            args.act_type,
            args.dropout,
            args.sig,
            args.use_sample,
            args.pa,
            args.gt
        ).cuda()

    elif args.model_type == "NRI":

        encoder_type = 'mlp'
        decoder_type = 'rnn'
        D_hidden_enc = 256
        D_hidden_dec = 256
        edge_types = args.heads_num  # Not a good design choice!

        model = NRI(encoder_type, decoder_type, system.state_num, D_hidden_enc, D_hidden_dec, edge_types, args.agent_num, args.input_length, args.ww, factor=True, dropout=0.0, skip_first=False, extend=False).cuda()
    
    elif args.model_type == "JointLSTM":

        D_hidden = 128
        if args.system == "motion":
            D_hidden = 64
        model = JointLSTM(system.state_num, D_hidden, system.state_num - system.const_num,
                          args.agent_num, 2, 0., args.ww).cuda()

    elif args.model_type == "SingleLSTM":
        D_hidden = 128
        if args.system == "motion":
            D_hidden = 64
        model = SingleLSTM(system.state_num, D_hidden, system.state_num - system.const_num,
                          args.agent_num, 2, 0., args.ww).cuda()

    else:
        raise NotImplementedError

    model_train = model.model_train
    model_test = model.model_test

    var_counts = count_vars(model)
    if rank_0:
        logger.log("NN architecture: %s" % args.model_type)
        logger.log("\nNumber of parameters: %d\n" % var_counts)

    # define loss function (criterion) and optimizer

    criterion = gmm_criterion(1)  # 2 answers, but no correlation presumed
    sampler = gmm_sample(1, r=True if args.sig else False)  # 2 answers, but no correlation presumed

    name_list = []
    if model.act_type == 'srelu':
        name_list = ['SR.tl', 'SR.tr', 'SR.ar']
    elif model.act_type == 'nrelu':
        name_list = ['norm_param']
    else:
        pass
    params = group_weight(model, name_list)
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=0.5, verbose=True if rank_0 else False) # No scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    if args.model_type == 'NRI':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)  # No scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5, verbose=True if rank_0 else False)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 1, eta_min=0)
    # scheduler = RLRP(optimizer, "min", factor=0.5, patience=20, min_lr=1e-8, verbose=1)

    if exp_name_cp:
        if rank_0:
            print(f"cp entered, {checkpoint_epoch}")
        rel_path = f'./result/runs/{exp_name_cp}/{exp_name_cp}_s{args.seed}/'
        checkpoint = torch.load(rel_path + f'pyt_save/model{checkpoint_epoch}.pth', map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        #scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"LR : {checkpoint['optimizer']['param_groups'][0]['lr']}")

        if checkpoint_lr:
            change_rate = float(checkpoint_lr)
            for g in optimizer.param_groups:
                g['initial_lr'] = change_rate
                g['lr'] = change_rate
                if local_rank == 0:
                    print(f'change rate : {change_rate}, changed lr : {g["lr"]}')
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, verbose=True if rank_0 else False)
            print(scheduler.state_dict())

        del checkpoint
        torch.cuda.empty_cache()

    # Parallelize model and start training

    if args.distributed:
        Parallel = DistributedDataParallel
        parallel_args = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "find_unused_parameters": True
        }
    else:
        Parallel = DataParallel
        parallel_args = {
            'device_ids': list(range(n_GPUs)),
            'output_device': 0
        }

    if args.system == 'spring':
        train_loader, test_loader, train_sampler, test_sampler, *data_stat = dataloader_spring(args.batch_size, f"spring_{args.agent_num}_t{args.dt}", data_folder='data', len_enc=args.input_length, len_dec=args.output_length, data_ratio=1.0, noise_var=args.noise_var, distributed=args.distributed)
                                                            
    elif args.system == 'kuramoto':
        train_loader, test_loader, train_sampler, test_sampler, *data_stat = dataloader_kuramoto(args.batch_size, f"kuramoto_{args.agent_num}_t{int(args.dt)}_{args.sample_mode}_{args.interaction_type}", data_folder='data', len_enc=args.input_length, len_dec=args.output_length, data_ratio=1.0, noise_var=args.noise_var, ww=args.ww, distributed=args.distributed)

    elif args.system == 'charge':
        train_loader, test_loader, train_sampler, test_sampler, *data_stat = dataloader_spring(args.batch_size, f"charge_{args.agent_num}_t{args.dt}", data_folder='data', len_enc=args.input_length, len_dec=args.output_length, data_ratio=1.0, noise_var=args.noise_var, distributed=args.distributed)

    elif args.system == 'motion':
        train_loader, test_loader, train_sampler, test_sampler, *data_stat = dataloader_motion(args.batch_size, None, data_folder='data', len_enc=args.input_length, len_dec=args.output_length, data_ratio=1.0, noise_var=args.noise_var, distributed=args.distributed)

    else:
        raise NotImplementedError

    model = Parallel(model, **parallel_args)
    if rank_0:
        logger.setup_pytorch_saver({'model': model.module, 'optimizer': optimizer, 'scheduler': scheduler})
    train_loss, train_count, test_loss, test_count = None, None, None, None
    start_time = time.time()
    start_epoch = args.start_epoch
    epochs = args.epochs
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        # train for one epoch
        if local_rank == 0:
            print("============== Epoch {} =============".format(epoch))
        train_start_time = time.time()
        train_loss, train_count = model_train(train_loader, model, criterion, optimizer, epoch, sampler, local_rank, args)
        if local_rank == 0:
            print(f'total train time : {time.time() - train_start_time}')

        if epoch > args.forcing_period:
            if scheduler is not None:
                scheduler.step()
        if rank_0:
            if args.model_type in ["NRI", "JointLSTM", 'SingleLSTM']:
                logger.store(TrainLoss_0=-1, TrainLoss_10=-1, TrainLoss_20=-1, TrainLoss_30=-1, TrainLoss_40=-1, TrainLoss_50=-1, TrainLoss_total=train_loss[-1])
            else:
                logger.store(TrainLoss_0=train_loss[0], TrainLoss_10=train_loss[9], TrainLoss_20=train_loss[19], TrainLoss_30=train_loss[29], TrainLoss_40=train_loss[39], TrainLoss_50=train_loss[49], TrainLoss_total=train_loss[-1])

        # evaluate on test set
        if epoch % test_freq == 0:
            test_loss, test_count, output = model_test(test_loader, model, criterion, sampler, local_rank, args)
            if rank_0:
                if args.model_type in ["NRI", "JointLSTM", 'SingleLSTM']:
                    logger.store(TestLoss_0=-1, TestLoss_10=-1, TestLoss_20=-1, TestLoss_30=-1, TestLoss_40=-1, TestLoss_50=-1,TestLoss_total=test_loss[-1])
                else:
                    logger.store(TestLoss_0=test_loss[0], TestLoss_10=test_loss[9], TestLoss_20=test_loss[19], TestLoss_30=test_loss[29], TestLoss_40=test_loss[39], TestLoss_50=test_loss[49], TestLoss_total=test_loss[-1])

        # Save model
        if rank_0:
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("Time", time.time() - start_time)
            logger.log_tabular("TrainLoss_0", average_only=True)
            logger.log_tabular("TrainLoss_10", average_only=True)
            logger.log_tabular("TrainLoss_20", average_only=True)
            logger.log_tabular("TrainLoss_30", average_only=True)
            logger.log_tabular("TrainLoss_40", average_only=True)
            logger.log_tabular("TrainLoss_50", average_only=True)
            logger.log_tabular("TrainLoss_total", average_only=True)

            if epoch % test_freq == 0:
                best_model = (test_loss[-1] < best_test_loss)
                if best_model:
                    best_test_loss = test_loss[-1]
                    print(f'best model saving : Epoch {epoch}')
                    logger.save_state(itr=epoch)
                elif epoch % save_freq == 0:
                    print(f'regular model saving : Epoch {epoch}')
                    logger.save_state(itr=epoch)
                logger.log_tabular("TestLoss_0", average_only=True)
                logger.log_tabular("TestLoss_10", average_only=True)
                logger.log_tabular("TestLoss_20", average_only=True)
                logger.log_tabular("TestLoss_30", average_only=True)
                logger.log_tabular("TestLoss_40", average_only=True)
                logger.log_tabular("TestLoss_50", average_only=True)
                logger.log_tabular("TestLoss_total", average_only=True)
            logger.dump_tabular()

            # Figure drawing
            draw_figure = False if args.system == 'motion' else True
            if draw_figure and (epoch > args.forcing_period) and (epoch % test_freq == 0):
                data_enc, data_dec, edges, attention_score, weight, prediction_list, test_loss_list = output
                if args.system == 'spring' or args.system == 'charge':
                    print('spring / charge plot entered')
                    draw_num = 5
                    data_enc_plot = DCN(data_enc.squeeze(0).transpose(1, 0))
                    data_dec_plot = DCN(data_dec.squeeze(0).transpose(1, 0))
                    data_predict_plot = DCN(prediction_list)
                    fig = plt.figure(figsize=(8, 7), dpi=150)
                    ax1 = fig.add_subplot(221)
                    ax2 = fig.add_subplot(222)
                    ax3 = fig.add_subplot(223)
                    ax4 = fig.add_subplot(224)

                    pd_time = 50
                    for i in range(draw_num):
                        ax1.scatter(data_enc_plot[i][:, 0], data_enc_plot[i][:, 1], c=color_list[i], marker='o', s=30, lw=0, alpha=0.1)
                    for i in range(draw_num):
                        ax1.scatter(data_dec_plot[i][:pd_time, 0], data_dec_plot[i][:pd_time, 1], c=color_list[i], marker='o', s=30, label='ball '+str(i), alpha=1)
                        ax1.legend(loc=4, prop={'size': 10})
                        ax1.scatter(data_dec_plot[i][pd_time - 1, 0], data_dec_plot[i][pd_time - 1, 1], c=color_list[i], marker='o', s=120)

                    for i in range(draw_num):
                        ax2.scatter(data_enc_plot[i][:, 0], data_enc_plot[i][:, 1], c=color_list[i], marker='o', s=30, alpha=0.1)
                    for i in range(draw_num):
                        ax2.scatter(data_predict_plot[i][:pd_time, 0], data_predict_plot[i][:pd_time, 1], c=color_list[i], marker='o', s=30, alpha=1)
                        ax2.scatter(data_predict_plot[i][pd_time - 1, 0], data_predict_plot[i][pd_time - 1, 1], c=color_list[i], marker='o', s=120)

                    if args.model_type not in ["JointLSTM", 'SingleLSTM']:  # no edges
                        im1 = ax3.imshow(edges, vmin=np.min(edges), vmax=np.max(edges))
                        im2 = ax4.imshow(weight, vmin=np.min(weight), vmax=np.max(weight))
                        fig.colorbar(im1, ax=ax3, fraction=0.046)
                        fig.colorbar(im2, ax=ax4, fraction=0.046)
                    ax1.set_title('Groundtruth')
                    ax2.set_title('RAIN Predcition')
                    plt.tight_layout()

                elif args.system == 'kuramoto':
                    print('kuramoto plot entered')
                    data_predict_plot = DCN(prediction_list)
                    data_enc_plot = DCN(data_enc.squeeze(0).transpose(1, 0))
                    data_dec_plot = DCN(data_dec.squeeze(0).transpose(1, 0))
                    data_enc_plot_gt = np.concatenate((data_enc_plot, np.expand_dims(data_dec_plot[:, 0, :], axis=1)), axis=1)
                    data_enc_plot_pd = np.concatenate((data_enc_plot, np.expand_dims(data_predict_plot[:, 0, :], axis=1)), axis=1)

                    fig = plt.figure(figsize=(8, 4), dpi=150)
                    ax1 = fig.add_subplot(521)
                    ax2 = fig.add_subplot(523)
                    ax3 = fig.add_subplot(525)
                    ax4 = fig.add_subplot(527)
                    ax5 = fig.add_subplot(529)
                    ax6 = fig.add_subplot(522)
                    ax7 = fig.add_subplot(524)
                    ax8 = fig.add_subplot(526)
                    ax9 = fig.add_subplot(528)
                    ax10 = fig.add_subplot(5, 2, 10)
                    x = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

                    enc_time = args.input_length
                    dec_time = args.output_length
                    val = 0

                    for i in range(5):
                        x[i].plot(np.arange(0, enc_time + 1), data_enc_plot_gt[i][:, val], c=color_list[i], lw=1, alpha=0.1, marker='o', ms=5)
                        x[i].plot(np.arange(enc_time + 1, enc_time + dec_time + 1), data_dec_plot[i][:, val], c=color_list[i], label='ball ' + str(i + 1), alpha=1, marker='o', ms=5)
                        if i != 4:
                            x[i].tick_params(axis='x', labelleft=False, labelbottom=False)
                        
                    for i in range(5):
                        x[i + 5].plot(np.arange(0, enc_time + 1), data_enc_plot_pd[i][:, val], c=color_list[i], lw=1, alpha=0.1, marker='o', ms=5)
                        x[i + 5].plot(np.arange(enc_time + 1, enc_time + dec_time + 1), data_predict_plot[i][:, val], c=color_list[i], label='ball ' + str(i + 1), alpha=1, marker='o', ms=5)
                        
                    x[0].set_title('Groundtruth', fontsize=20)
                    x[5].set_title('RAIN Predcition', fontsize=20)
                    plt.tight_layout()
                elif args.system == 'motion':
                    pass
                else:
                    raise NotImplementedError
                
                logger.writer.add_figure('Average Performance', fig, global_step=epoch)
                print('finish')

    if rank_0:
        logger.close()


if __name__ == "__main__":
    print("started!")  # For test
    main()
