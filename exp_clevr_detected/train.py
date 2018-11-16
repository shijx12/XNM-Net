import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to import shared utils
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import shutil
import copy

from DataLoader import ClevrDataLoader
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
        'feature_pt': args.train_feature_pt,
        'vocab_json': args.vocab_json,
        'batch_size': args.batch_size,
        'edge_class': args.edge_class,
        'ratio': args.ratio,
        'shuffle': True,
    }
    val_loader_kwargs = {
        'question_pt': args.val_question_pt,
        'feature_pt': args.val_feature_pt,
        'vocab_json': args.vocab_json,
        'batch_size': args.batch_size,
        'edge_class': args.edge_class,
        'shuffle': False,
    }
    
    train_loader = ClevrDataLoader(**train_loader_kwargs)
    val_loader = ClevrDataLoader(**val_loader_kwargs)

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = { k:v for k,v in vars(args).items() if k in {
        'dim_v', 'dim_edge', 'dim_feature', 'k_attr', 'num_edge_cat', 'num_class', 'edge_class',
        } }
    model_kwargs_tosave = copy.deepcopy(model_kwargs) 
    model_kwargs['vocab'] = train_loader.vocab
    model = XNMNet(**model_kwargs).to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.l2reg)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(1/args.ratio)], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    logging.info("Start training........")
    iter_count = 0
    for epoch in range(args.num_epoch):
        for i, batch in enumerate(train_loader):
            iter_count += 1
            progress = epoch+i/len(train_loader)
            answers, questions, *batch_input = [todevice(x, device) for x in batch]
            logits, others = model(*batch_input)
            loss = criterion(logits, answers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % (len(train_loader) // 10) == 0:
                logging.info("Progress %.3f  loss = %.3f" % (progress, loss.item()))
        scheduler.step()
        if (epoch+1) % 1 == 0:
            valid_acc = validate(model, val_loader, device)
            logging.info('\n ~~~~~~ Valid Accuracy: %.3f ~~~~~~~\n' % valid_acc)

            save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(args.save_dir, 'model.pt')) 
            logging.info(' >>>>>> save to %s <<<<<<' % (args.save_dir))


def save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs_tosave,
        }
    torch.save(state, filename)



def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--save_dir', type=str, required=True, help='path to save checkpoints and logs')
    parser.add_argument('--input_dir', default='/data1/jiaxin/exp/CLEVR/data/')
    parser.add_argument('--train_question_pt', default='train_questions.pt')
    parser.add_argument('--train_feature_pt', default='train_features_salient_thres0.4.pt')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--val_feature_pt', default='val_features_salient_thres0.4.pt')
    parser.add_argument('--vocab_json', default='vocab.json')
    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2reg', default=5e-6, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--ratio', default=1, type=float, help='ratio of training examples')
    # model hyperparameters
    parser.add_argument('--dim_v', default=128, type=int)
    parser.add_argument('--dim_feature', default=512, type=int)
    parser.add_argument('--dim_edge', default=2, type=int)
    parser.add_argument('--k_attr', default=4, type=int)
    parser.add_argument('--num_edge_cat', default=5, type=int)
    parser.add_argument('--num_class', default=28, type=int)
    parser.add_argument('--edge_class', choices=['rule', 'learncat', 'dense'], default='dense')
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
    args.train_feature_pt = os.path.join(args.input_dir, args.train_feature_pt)
    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.val_question_pt)
    args.val_feature_pt = os.path.join(args.input_dir, args.val_feature_pt)
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
