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
        coco_ids, answers, *params = [todevice(x, device) for x in batch]
        logits, others = model(*params)
        acc = batch_accuracy(logits, answers)
        total_acc += acc.sum().item()
        count += answers.size(0)
    acc = total_acc / count
    return acc


def visualize(model, data, device):
    model.eval()
    for batch in data:
        coco_ids, answers, *params = [todevice(x, device) for x in batch]
        predict_str, intermediates = model.forward_and_return_intermediates(*params)
        if intermediates is None:
            continue
        answer_str = data.vocab['answer_idx_to_token'][answers[0].max(0)[1].item()]
        if True: #predict_str != answer_str:
            questions = params[0]
            question_str = ' '.join(list(filter(lambda x: x!='<NULL>', [data.vocab['question_idx_to_token'][q.item()] for q in questions[0]])))

            print("="*88)
            desc = data.scene_descs[coco_ids[0]]
            for _ in intermediates:
                if _ is None:
                    continue
                module_type, node_attn = _
                print(" >> %s" % module_type)
                if node_attn is None:
                    continue
                for weight, node_desc in zip(node_attn[:len(desc)], desc):
                    print("%.3f\t%s" % (weight, node_desc))
                print(" <<<<<< ")
            print("question: %s" % question_str)
            print("answer: %s, predict: %s" % (answer_str, predict_str))
            embed()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='val', choices=['val', 'vis'])
    # input
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input_dir', default='/data/sjx/VQA-Exp/data')
    parser.add_argument('--data_type', choices=['coco', 'vg'])
    parser.add_argument('--train_question_pt', default='train_questions.pt')
    parser.add_argument('--train_scene_pt', default='train_sg.pt')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--val_scene_pt', default='val_sg.pt')
    parser.add_argument('--vocab_json', default='vocab.json')
    parser.add_argument('--feature_h5', default='trainval_feature.h5')
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int, help='hidden state of seq2seq parser')
    parser.add_argument('--dim_v', default=300, type=int, help='node/attribute/edge embedding')
    parser.add_argument('--dim_vision', default=2048, type=int)
    parser.add_argument('--cls_fc_dim', default=1024, type=int)
    parser.add_argument('--class_mode', default='qvc', choices=['qvc', 'qv', 'c'])
    parser.add_argument('--dropout', default=0.5, type=float)
    args = parser.parse_args()

    args.train_question_pt = os.path.join(args.input_dir, args.data_type+'_'+args.train_question_pt)
    args.train_scene_pt = os.path.join(args.input_dir, args.data_type+'_'+args.train_scene_pt)
    args.vocab_json = os.path.join(args.input_dir, args.data_type+'_'+args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.data_type+'_'+args.val_question_pt)
    args.val_scene_pt = os.path.join(args.input_dir, args.data_type+'_'+args.val_scene_pt)
    args.feature_h5 = os.path.join(args.input_dir, args.feature_h5)
    
    device = 'cuda' if args.mode == 'val' else 'cpu'
    val_loader_kwargs = {
        'question_pt': args.val_question_pt,
        'scene_pt': args.val_scene_pt,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': 128 if args.mode == 'val' else 1,
        'shuffle': False
    }
    train_loader_kwargs = {
        'question_pt': args.train_question_pt,
        'scene_pt': args.train_scene_pt,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': 128 if args.mode == 'val' else 1,
        'shuffle': False
    }
    train_loader = VQADataLoader(**train_loader_kwargs)
    val_loader = VQADataLoader(**val_loader_kwargs)
    model_kwargs = {
        'vocab': val_loader.vocab,
        'dim_v': args.dim_v,
        'dim_hidden': args.dim_hidden,
        'dim_vision': args.dim_vision,
        'class_mode': args.class_mode,
        'dropout_prob': args.dropout,
        'device': device,
        'cls_fc_dim': args.cls_fc_dim,
        'program_scheme': ['find', 'relate', 'describe'],
    }
    model = XNMNet(**model_kwargs).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})['state_dict'])

    if args.mode == 'val':
        train_acc = validate(model, train_loader, device)
        print('train acc: %.4f' % train_acc)
        valid_acc = validate(model, val_loader, device)
        print('valid acc: %.4f' % valid_acc)
    else:
        visualize(model, train_loader, device)
        visualize(model, val_loader, device)
    # cmd: python3.6 validate.py --ckpt /data/sjx/CLEVR-Exp/acc100/model.pt


