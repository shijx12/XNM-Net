import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils.clevr import ClevrDataLoader
from tbd.module_net import TbDNet
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
from validate import validate, show_edge_attention


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
    
    train_loader = ClevrDataLoader(**train_loader_kwargs)
    val_loader = ClevrDataLoader(**val_loader_kwargs)

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'vocab': train_loader.vocab,
        'dim_pre_v': args.dim_pre_v,
        'dim_v': args.dim_v,
    }
    model = TbDNet(**model_kwargs).to(device)
    logging.info(model)
    if args.ckpt and os.path.isfile(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})['state_dict'])
        for name, param in model.named_parameters():
            # if name not in {'edge_cat_vectors', 'word_embedding.weight'}:
            if name not in {'word_embedding.weight'}:
                param.requires_grad = False
    else:
        print('ckpt is not specified')
    parameters = [param for param in model.parameters() if param.requires_grad]

    optimizer = optim.Adam(parameters, args.lr, weight_decay=args.l2reg)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1], gamma=args.lr_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    logging.info("Start training........")
    writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
    tic = time.time()
    iter_count = 0
    for epoch in range(args.num_epoch):
        for i, batch in enumerate(train_loader.generator()):
            iter_count += 1
            progress = epoch+i/len(train_loader)
            _, answers, programs, program_inputs, conn_matrixes, cat_matrixes, pre_v = \
                    [todevice(x, device) for x in batch]

            logits = model(programs, program_inputs, conn_matrixes, cat_matrixes, pre_v)
            loss = criterion(logits, answers)
            regular = torch.abs(torch.eye(9).to(device) - torch.matmul(model.edge_cat_vectors, model.edge_cat_vectors.t())).sum()
            loss += args.regular_weight * regular
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                logging.info("Progress %.3f  loss = %.3f" % (progress, loss.item()))
                show_edge_attention(model, val_loader)
            if (i+1) % 10 == 0:
                writer.add_scalar('loss', loss, iter_count)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), iter_count)
                        if param.grad is not None:
                            writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), iter_count)
            if args.val and (i+1) % 2000 == 0:
                valid_acc = validate(model, val_loader, device)
                writer.add_scalar('valid_acc', valid_acc, iter_count)
                logging.info('\n==================\n Valid Accuracy: %.3f \n==================' % valid_acc)
            if iter_count == 1500:
                scheduler.step()
                print('scheduler step')
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
    parser.add_argument('--ckpt')
    parser.add_argument('--input_dir', default='/data/sjx/CLEVR-Exp/data')
    parser.add_argument('--train_question_pt', default='train_questions.pt')
    parser.add_argument('--train_scene_pt', default='train_scenes.pt')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--val_scene_pt', default='val_scenes.pt')
    parser.add_argument('--save_dir', type=str, required=True, help='path to save checkpoints and logs')
    parser.add_argument('--vocab_json', default='vocab.json')
    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--clip', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=3, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--regular_weight', default=0.001, type=float)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--val', action='store_true')
    # model hyperparameters
    parser.add_argument('--dim_pre_v', default=15, type=int)
    parser.add_argument('--dim_v', default=128, type=int)
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
