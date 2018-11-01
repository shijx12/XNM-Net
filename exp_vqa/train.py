import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to import shared utils
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import shutil
from IPython import embed

from DataLoader import VQADataLoader
from model.net import XNMNet
from utils.misc import todevice
from validate import validate

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()


def train(args):
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
        'question_pt': args.train_question_pt,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': args.batch_size,
        'spatial': args.spatial,
        'num_workers': 1,
        'shuffle': True,
    }
    val_loader_kwargs = {
        'question_pt': args.val_question_pt,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': args.batch_size,
        'spatial': args.spatial,
        'num_workers': 3,
        'shuffle': False,
    }
    
    train_loader = VQADataLoader(**train_loader_kwargs)
    val_loader = VQADataLoader(**val_loader_kwargs)

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'vocab': train_loader.vocab,
        'dim_v': args.dim_v,
        'dim_edge': args.dim_edge,
        'dim_word': args.dim_word,
        'dim_hidden': args.dim_hidden,
        'dim_vision': args.dim_vision,
        'class_mode': args.class_mode,
        'dropout_prob': args.dropout,
        'device': device,
        'spatial': args.spatial,
        'cls_fc_dim': args.cls_fc_dim,
        'program_scheme': ['find', 'relate', 'describe'],
    }
    model_kwargs_tosave = { k:v for k,v in model_kwargs.items() if k != 'vocab' }
    model = XNMNet(**model_kwargs).to(device)
    logging.info(model)
    logging.info('load glove vectors')
    train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)
    model.token_embedding.weight.data.set_(train_loader.glove_matrix)
    ################################################################

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, args.lr, weight_decay=args.l2reg)

    start_epoch = 0
    if args.restore:
        print("Restore checkpoint and optimizer...")
        ckpt = os.path.join(args.save_dir, 'model.pt')
        ckpt = torch.load(ckpt, map_location={'cuda:0': 'cpu'})
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5**(1 / args.lr_halflife))
    
    logging.info("Start training........")
    #print(validate(model, val_loader, device))
    for epoch in range(start_epoch, args.num_epoch):
        model.train()
        for i, batch in enumerate(train_loader):
            progress = epoch+i/len(train_loader)
            coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            logits, others = model(*batch_input)
            ##################### loss #####################
            nll = -nn.functional.log_softmax(logits, dim=1)
            loss = (nll * answers / 10).sum(dim=1).mean()
            #################################################
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(parameters=parameters, clip_value=0.5)
            optimizer.step()

            if (i+1) % (len(train_loader) // 100) == 0:
                logging.info("Progress %.3f  ce_loss = %.3f" % (progress, loss.item()))
                #print(progress)
        valid_acc = validate(model, val_loader, device)
        logging.info('\n ~~~~~~ Valid Accuracy: %.4f ~~~~~~~\n' % valid_acc)
        save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(args.save_dir, 'model.pt')) 
        logging.info(' >>>>>> save to %s <<<<<<' % (args.save_dir))


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
        }
    torch.save(state, filename)



def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--save_dir', type=str, required=True, help='path to save checkpoints and logs')
    parser.add_argument('--input_dir', default='/data/sjx/VQA-Exp/data')
    #parser.add_argument('--input_dir', default='/data1/jiaxin/exp/vqa/data')
    parser.add_argument('--data_type', default='coco', choices=['coco', 'vg'])
    parser.add_argument('--train_question_pt', default='train_questions.pt')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--vocab_json', default='vocab.json')
    parser.add_argument('--feature_h5', default='trainval_feature.h5')
    parser.add_argument('--restore', action='store_true')
    # training parameters
    parser.add_argument('--lr', default=1.5e-3, type=float)
    parser.add_argument('--lr_halflife', default=50000, type=int)
    parser.add_argument('--l2reg', default=0, type=float)
    parser.add_argument('--num_epoch', default=150, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int, help='dim of word/node/attribute/edge embedding')
    parser.add_argument('--dim_hidden', default=1024, type=int, help='hidden state of seq2seq parser')
    parser.add_argument('--dim_v', default=512, type=int, help='dim of node embedding')
    parser.add_argument('--dim_edge', default=256, type=int, help='dim of edge embedding')
    parser.add_argument('--dim_vision', default=2048, type=int)
    parser.add_argument('--cls_fc_dim', default=1024, type=int)
    parser.add_argument('--class_mode', default='qvc', choices=['qvc', 'qv', 'c'])
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--spatial', action='store_true')
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not args.restore:
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        os.mkdir(args.save_dir)
    else:
        assert os.path.isdir(args.save_dir)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))
    # concat obsolute path of input files
    args.train_question_pt = os.path.join(args.input_dir, args.data_type+'_'+args.train_question_pt)
    args.vocab_json = os.path.join(args.input_dir, args.data_type+'_'+args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.data_type+'_'+args.val_question_pt)
    args.feature_h5 = os.path.join(args.input_dir, args.feature_h5)
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.spatial:
        args.dim_vision += 5

    train(args)


if __name__ == '__main__':
    main()
