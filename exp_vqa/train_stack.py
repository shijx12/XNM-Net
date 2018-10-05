import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils.vqa import VQADataLoader
from model_stack.module_net import TbDNet
import argparse
import time
import os
import shutil
from tensorboardX import SummaryWriter
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
from utils.misc import todevice
from validate import validate
import pickle


def train(args):
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
        'question_pt': args.train_question_pt,
        'scene_pt': args.train_scene_pt,
        'vocab_json': args.vocab_json,
        'batch_size': args.batch_size,
        'shuffle': True
    }
    val_loader_kwargs = {
        'question_pt': args.val_question_pt,
        'scene_pt': args.val_scene_pt,
        'vocab_json': args.vocab_json,
        'batch_size': args.batch_size,
        'shuffle': False
    }
    
    train_loader = VQADataLoader(**train_loader_kwargs)
    val_loader = VQADataLoader(**val_loader_kwargs)

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'vocab': train_loader.vocab,
        'dim_v': args.dim_v,
        'dim_word': args.dim_word,
        'dim_hidden': args.dim_hidden,
        'cls_fc_dim': args.cls_fc_dim,
        'dropout': args.dropout,
        'T_ctrl': args.T_ctrl,
        'stack_len': args.stack_len,
        'device': device,
        'use_gumbel': args.module_prob_use_gumbel==1,
        'use_validity': args.module_prob_use_validity==1,
    }
    model = TbDNet(**model_kwargs).to(device)
    logging.info(model)
    logging.info('load glove vectors')
    train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)
    model.token_embedding.weight.data.set_(train_loader.glove_matrix)
    ################################################################

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.l2reg)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[args.lr_decay_stone], gamma=args.lr_decay)
    logging.info("Start training........")
    writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
    tic = time.time()
    iter_count = 0
    
    # print(validate(model, val_loader, device))
    for epoch in range(args.num_epoch):
        model.train()
        for i, batch in enumerate(train_loader.generator()):
            iter_count += 1
            progress = epoch+i/len(train_loader)
            questions, questions_len, gt_programs, answers, conn_matrixes, cat_matrixes, vertex_indexes = \
                    [todevice(x, device) for x in batch]

            logits, others = model(questions, questions_len, conn_matrixes, cat_matrixes, vertex_indexes)
            ##################### loss #####################
            ce_loss = criterion(logits, answers)
            # layout_loss = torch.mean(-others['log_seq_prob']) # gt: layout_loss; rl: policy_loss
            # entropy_loss = args.lambda_entropy * torch.mean(others['neg_entropy'])
            # policy_loss = torch.mean((ce_loss_all.detach() - policy_gradient_baseline) * others['log_seq_prob']) # element-wise multiply
            # num_edge = len(train_loader.vocab['edge_token_to_idx'])
            # regular_loss = torch.abs(torch.eye(num_edge).to(device) - torch.matmul(model.edge_cat_vectors, model.edge_cat_vectors.t())).sum()
            loss = args.lambda_answer * ce_loss #  + args.lambda_layout * layout_loss + args.lambda_regular * regular_loss
            #################################################
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip)
            optimizer.step()

            if (i+1) % (len(train_loader) // 200) == 0:
            #if True:
                # logging.info("Progress %.3f  ce_loss = %.3f, layout_loss = %.3f" % (progress, ce_loss.item(), layout_loss.item()))
                logging.info("Progress %.3f  ce_loss = %.3f" % (progress, ce_loss.item()))
            if (i+1) % (len(train_loader)) == 0:
            #if True:
                for name, param in model.named_parameters():
                    try:
                        writer.add_histogram(name, param, iter_count)
                        if param.grad is not None:
                            writer.add_histogram(name+'/grad', param.grad, iter_count)
                    except Exception as e:
                        print(name)

            # writer.add_scalar('loss', loss, iter_count)
        valid_acc = validate(model, val_loader, device)
        writer.add_scalar('valid_acc', valid_acc, iter_count)
        logging.info('\n==================\n Valid Accuracy: %.3f \n==================' % valid_acc)
        scheduler.step()
        save_checkpoint(epoch, model, optimizer, os.path.join(args.save_dir, 'model.pt')) 
        logging.info(' >>>>>> save model of EPOCH %d <<<<<<' % (epoch+1))


def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    torch.save(state, filename)



def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default='/data/sjx/VQA-Exp/data')
    parser.add_argument('--train_question_pt', default='train_questions.pt')
    parser.add_argument('--train_scene_pt', default='train_scenes.pt')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--val_scene_pt', default='val_scenes.pt')
    parser.add_argument('--save_dir', type=str, required=True, help='path to save checkpoints and logs')
    parser.add_argument('--vocab_json', default='vocab.json')
    # training parameters
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--lr_decay_stone', default=3, type=int)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--clip', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=7, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    # loss lambda
    parser.add_argument('--lambda_answer', default=1, type=float)
    parser.add_argument('--lambda_regular', default=0.001, type=float)
    parser.add_argument('--lambda_layout', default=1, type=float)
    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int, help='word embedding')
    parser.add_argument('--dim_hidden', default=600, type=int, help='hidden state of seq2seq parser')
    parser.add_argument('--dim_v', default=512, type=int, help='node/attribute/edge embedding')
    parser.add_argument('--cls_fc_dim', default=1024, type=int, help='classifier fc dim')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--T_ctrl', default=10, type=int, help='controller decode length')
    parser.add_argument('--stack_len', default=4, type=int, help='stack length')
    parser.add_argument('--module_prob_use_gumbel', default=1, choices=[0, 1], type=int, help='whether use gumbel softmax for module prob. 0 not use, 1 use')
    parser.add_argument('--module_prob_use_validity', default=0, choices=[0, 1], type=int, help='whether validate module prob.')
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))
    # concat obsolute path of input files
    args.train_question_pt = os.path.join(args.input_dir, args.train_question_pt)
    args.train_scene_pt = os.path.join(args.input_dir, args.train_scene_pt)
    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.val_question_pt)
    args.val_scene_pt = os.path.join(args.input_dir, args.val_scene_pt)
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
