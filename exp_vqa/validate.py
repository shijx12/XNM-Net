import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to import shared utils
import torch
from tqdm import tqdm
import argparse
import numpy as np
import os
import json
from IPython import embed

from DataLoader import VQADataLoader
from model.net import XNMNet
from model_stack.net import XNMNet as StackXNMNet
from utils.misc import todevice


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = true.gather(dim=1, index=predicted_index)
    '''
    Acc needs to be averaged over all 10 choose 9 subsets of human answers.
    While we could just use a loop, surely this can be done more efficiently (and indeed, it can).
    There are two cases for the 1 chosen answer to be discarded:
    (1) the discarded answer is not the predicted answer => acc stays the same
    (2) the discarded answer is the predicted answer => we have to subtract 1 from the number of agreeing answers
    
    There are (10 - num_agreeing_answers) of case 1 and num_agreeing_answers of case 2, thus
    acc = ((10 - agreeing) * min( agreeing      / 3, 1)
           +     agreeing  * min((agreeing - 1) / 3, 1)) / 10
    
    Let's do some more simplification:
    if num_agreeing_answers == 0:
        acc = 0  since the case 1 min term becomes 0 and case 2 weighting term is 0
    if num_agreeing_answers >= 4:
        acc = 1  since the min term in both cases is always 1
    The only cases left are for 1, 2, and 3 agreeing answers.
    In all of those cases, (agreeing - 1) / 3  <  agreeing / 3  <=  1, so we can get rid of all the mins.
    By moving num_agreeing_answers from both cases outside the sum we get:
        acc = agreeing * ((10 - agreeing) + (agreeing - 1)) / 3 / 10
    which we can simplify to:
        acc = agreeing * 0.3
    Finally, we can combine all cases together with:
        min(agreeing * 0.3, 1)
    '''
    return (agreeing * 0.3).clamp(max=1)


def validate(model, data, device):
    count, correct = 0, 0
    model.eval()
    print('validate...')
    total_acc, count = 0, 0
    for batch in tqdm(data, total=len(data)):
        coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]
        logits, others = model(*batch_input)
        acc = batch_accuracy(logits, answers)
        total_acc += acc.sum().item()
        count += answers.size(0)
    acc = total_acc / count
    return acc


def visualize(model, data, device):
    model.eval()
    for batch in data:
        coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]
        batch_input.append(True)
        logits, others = model(*batch_input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='vis', choices=['val', 'vis'])
    parser.add_argument('--model', default='stack', choices=['stack', 'seq'])
    # input
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input_dir', default='/data/sjx/VQA-Exp/data')
    parser.add_argument('--data_type', default='coco', choices=['coco', 'vg'])
    parser.add_argument('--train_question_pt', default='train_questions.pt')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--vocab_json', default='vocab.json')
    parser.add_argument('--feature_h5', default='trainval_feature.h5')
    args = parser.parse_args()

    args.train_question_pt = os.path.join(args.input_dir, args.data_type+'_'+args.train_question_pt)
    args.vocab_json = os.path.join(args.input_dir, args.data_type+'_'+args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.data_type+'_'+args.val_question_pt)
    args.feature_h5 = os.path.join(args.input_dir, args.feature_h5)
    
    device = 'cuda' if args.mode == 'val' else 'cpu'
    loaded = torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})
    model_kwargs = loaded['model_kwargs']
    val_loader_kwargs = {
        'question_pt': args.val_question_pt,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': 128 if args.mode == 'val' else 1,
        'spatial': model_kwargs['spatial'],
        'shuffle': False
    }
    train_loader_kwargs = {
        'question_pt': args.train_question_pt,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': 128 if args.mode == 'val' else 1,
        'spatial': model_kwargs['spatial'],
        'shuffle': False
    }
    train_loader = VQADataLoader(**train_loader_kwargs)
    val_loader = VQADataLoader(**val_loader_kwargs)

    model_kwargs.update({'vocab': val_loader.vocab, 'device': device})
    Model = XNMNet if args.model == 'seq' else StackXNMNet
    model = Model(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'])

    if args.mode == 'val':
        train_acc = validate(model, train_loader, device)
        print('train acc: %.4f' % train_acc)
        valid_acc = validate(model, val_loader, device)
        print('valid acc: %.4f' % valid_acc)
    else:
        visualize(model, val_loader, device)

