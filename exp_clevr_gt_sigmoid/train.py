import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to import shared utils
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import os
import copy
from IPython import embed

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
        'scene_pt': args.train_scene_pt,
        'vocab_json': args.vocab_json,
        'batch_size': args.batch_size,
        'ratio': args.ratio,
        'shuffle': True
    }
    val_loader_kwargs = {
        'question_pt': args.val_question_pt,
        'scene_pt': args.val_scene_pt,
        'vocab_json': args.vocab_json,
        'batch_size': args.batch_size,
        'shuffle': False
    }
    
    train_loader = ClevrDataLoader(**train_loader_kwargs)
    val_loader = ClevrDataLoader(**val_loader_kwargs)

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = { k:v for k,v in vars(args).items() if k in {
        'dim_v', 'num_class',
        } }
    model_kwargs_tosave = copy.deepcopy(model_kwargs) 
    model_kwargs['vocab'] = train_loader.vocab
    model = XNMNet(**model_kwargs).to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.l2reg)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(1/args.ratio)], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    logging.info("Start training........")
    tic = time.time()
    iter_count = 0
    for epoch in range(args.num_epoch):
        for i, batch in enumerate(train_loader.generator()):
            iter_count += 1
            progress = epoch+i/len(train_loader)
            answers, questions, *batch_input = \
                    [todevice(x, device) for x in batch]

            logits = model(*batch_input)
            loss = criterion(logits, answers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % (len(train_loader) // 10) == 0:
                logging.info("Progress %.3f  loss = %.3f" % (progress, loss.item()))
        scheduler.step()
        if (epoch+1) % 1 == 0:
            valid_acc = validate(model, val_loader, device)
            logging.info('\n ~~~~~~ Valid Accuracy: %.4f ~~~~~~~\n' % valid_acc)

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
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--train_question_pt', default='train_questions.pt')
    parser.add_argument('--train_scene_pt', default='train_scenes.pt')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--val_scene_pt', default='val_scenes.pt')
    parser.add_argument('--vocab_json', default='vocab.json')
    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--ratio', default=1, type=float, help='ratio of training examples')
    # model hyperparameters
    parser.add_argument('--dim_v', default=128, type=int)
    parser.add_argument('--num_class', default=28, type=int)
    args = parser.parse_args()

    # make logging.info display into both shell and file
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
