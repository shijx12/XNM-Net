import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to import shared utils
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import shutil
from tensorboardX import SummaryWriter
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
        'scene_pt': args.train_scene_pt,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'shuffle': True,
        'pin_memory': True,
    }
    val_loader_kwargs = {
        'question_pt': args.val_question_pt,
        'scene_pt': args.val_scene_pt,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'shuffle': False,
        'pin_memory': True,
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
        'dim_vision': args.dim_vision,
        'class_mode': args.class_mode,
        'dropout_prob': args.dropout,
        'device': device,
        'cls_fc_dim': args.cls_fc_dim,
        'program_scheme': ['find', 'relate', 'describe'],
    }
    model_kwargs_tosave = { k:v for k,v in model_kwargs.items() if k != 'vocab' }
    model = XNMNet(**model_kwargs).to(device)
    logging.info(model)
    logging.info('load glove vectors')
    train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)
    model.token_embedding.weight.data.set_(train_loader.glove_matrix)
    if args.fix_token_embedding:
        model.token_embedding.weight.requires_grad = False
    ################################################################

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, args.lr, weight_decay=args.l2reg)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5**(1 / args.lr_halflife))
    logging.info("Start training........")
    writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
    tic = time.time()
    iter_count = 0
    
    #print(validate(model, val_loader, device))
    for epoch in range(args.num_epoch):
        model.train()
        for i, batch in enumerate(train_loader):
            iter_count += 1
            progress = epoch+i/len(train_loader)
            coco_ids, answers, *params = [todevice(x, device) for x in batch]
            # questions, questions_len, conn_matrixes, cat_matrixes, vertex_indexes, edge_indexes, vision_feat
            logits, others = model(*params)
            ##################### loss #####################
            nll = -nn.functional.log_softmax(logits, dim=1)
            ce_loss = (nll * answers / 10).sum(dim=1).mean()
            # layout_loss = torch.mean(-others['log_seq_prob']) # gt: layout_loss; rl: policy_loss
            # entropy_loss = args.lambda_entropy * torch.mean(others['neg_entropy'])
            # policy_loss = torch.mean((ce_loss_all.detach() - policy_gradient_baseline) * others['log_seq_prob']) # element-wise multiply
            loss = ce_loss
            #################################################
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(parameters=parameters, max_norm=args.clip)
            optimizer.step()

            if (i+1) % (len(train_loader) // 100) == 0:
                logging.info("Progress %.3f  ce_loss = %.3f" % (progress, ce_loss.item()))
                #print(progress)
            if (i+1) % (len(train_loader)) == 0:
                for name, param in model.named_parameters():
                    try:
                        writer.add_histogram(name, param, iter_count)
                        if param.grad is not None:
                            writer.add_histogram(name+'/grad', param.grad, iter_count)
                    except Exception as e:
                        print(name)
        valid_acc = validate(model, val_loader, device)
        writer.add_scalar('valid_acc', valid_acc, iter_count)
        logging.info('\n==================\n Valid Accuracy: %.3f \n==================' % valid_acc)
        save_checkpoint(epoch, model, model_kwargs_tosave, os.path.join(args.save_dir, 'model.pt')) 
        logging.info(' >>>>>> save to %s <<<<<<' % (args.save_dir))


def save_checkpoint(epoch, model, model_kwargs, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'model_kwargs': model_kwargs,
        }
    torch.save(state, filename)



def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default='/data/sjx/VQA-Exp/data')
    parser.add_argument('--data_type', choices=['coco', 'vg'])
    #parser.add_argument('--input_dir', default='/data1/jiaxin/exp/vqa/data')
    parser.add_argument('--train_question_pt', default='train_questions.pt')
    parser.add_argument('--train_scene_pt', default='train_sg.pt')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--val_scene_pt', default='val_sg.pt')
    parser.add_argument('--vocab_json', default='vocab.json')
    parser.add_argument('--feature_h5', default='trainval_feature.h5')
    parser.add_argument('--save_dir', type=str, required=True, help='path to save checkpoints and logs')
    # training parameters
    parser.add_argument('--lr', default=1.5e-3, type=float)
    parser.add_argument('--lr_halflife', default=50000, type=int)
    parser.add_argument('--l2reg', default=1e-6, type=float)
    parser.add_argument('--clip', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--fix_token_embedding', action='store_true')
    # loss lambda
    parser.add_argument('--lambda_answer', default=1, type=float)
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int, help='hidden state of seq2seq parser')
    parser.add_argument('--dim_word', default=300, type=int, help='dim of word/node/attribute/edge embedding')
    parser.add_argument('--dim_v', default=1024, type=int, help='dim of word/node/attribute/edge embedding')
    parser.add_argument('--dim_vision', default=2048, type=int)
    parser.add_argument('--cls_fc_dim', default=1024, type=int)
    parser.add_argument('--class_mode', default='qvc', choices=['qvc', 'qv', 'c'])
    parser.add_argument('--dropout', default=0.5, type=float)
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
    args.train_question_pt = os.path.join(args.input_dir, args.data_type+'_'+args.train_question_pt)
    args.train_scene_pt = os.path.join(args.input_dir, args.data_type+'_'+args.train_scene_pt)
    args.vocab_json = os.path.join(args.input_dir, args.data_type+'_'+args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.data_type+'_'+args.val_question_pt)
    args.val_scene_pt = os.path.join(args.input_dir, args.data_type+'_'+args.val_scene_pt)
    args.feature_h5 = os.path.join(args.input_dir, args.feature_h5)
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
