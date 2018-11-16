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
        coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]
        logits, others = model(*batch_input)
        acc = batch_accuracy(logits, answers)
        total_acc += acc.sum().item()
        count += answers.size(0)
    acc = total_acc / count
    return acc


def test(model, data, device):
    model.eval()
    results = []
    for batch in tqdm(data, total=len(data)):
        coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]
        logits, others = model(*batch_input)
        predicts = torch.max(logits, dim=1)[1]
        for predict in predicts:
            results.append(data.vocab['answer_idx_to_token'][predict.item()])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='val', choices=['val', 'test'])
    # input
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input_dir', default='/data/sjx/VQA-Exp/data')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--test_question_pt', default='test_questions.pt')
    parser.add_argument('--vocab_json', default='vocab.json')
    parser.add_argument('--feature_h5', default='trainval_feature.h5')
    parser.add_argument('--output_file', help='used only in test mode')
    parser.add_argument('--test_question_json', help='path to v2_OpenEnded_mscoco_test2015_questions.json, used only in test mode')
    args = parser.parse_args()

    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.val_question_pt)
    args.test_question_pt = os.path.join(args.input_dir, args.test_question_pt)
    args.feature_h5 = os.path.join(args.input_dir, args.feature_h5)
    
    device = 'cuda'
    loaded = torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})
    model_kwargs = loaded['model_kwargs']
    if args.mode == 'val':
        val_loader_kwargs = {
            'question_pt': args.val_question_pt,
            'vocab_json': args.vocab_json,
            'feature_h5': args.feature_h5,
            'batch_size': 128,
            'spatial': model_kwargs['spatial'],
            'num_workers': 2,
            'shuffle': False
        }
        val_loader = VQADataLoader(**val_loader_kwargs)
        model_kwargs.update({'vocab': val_loader.vocab, 'device': device})
        model = XNMNet(**model_kwargs).to(device)
        model.load_state_dict(loaded['state_dict'])
        valid_acc = validate(model, val_loader, device)
        print('valid acc: %.4f' % valid_acc)
    elif args.mode == 'test':
        assert args.output_file and os.path.exists(args.test_question_json)
        test_loader_kwargs = {
            'question_pt': args.test_question_pt,
            'vocab_json': args.vocab_json,
            'feature_h5': args.feature_h5,
            'batch_size': 128,
            'spatial': model_kwargs['spatial'],
            'num_workers': 2,
            'shuffle': False
        }
        test_loader = VQADataLoader(**test_loader_kwargs)
        model_kwargs.update({'vocab': test_loader.vocab, 'device': device})
        model = XNMNet(**model_kwargs).to(device)
        model.load_state_dict(loaded['state_dict'])
        results = test(model, test_loader, device)
        questions = json.load(open(args.test_question_json))['questions']
        assert len(results) == len(questions)
        results = [{'answer':r, "question_id":q['question_id']} for r,q in zip(results, questions)]
        with open(args.output_file, 'w') as f:
            json.dump(results, f)
        print('write into %s' % args.output_file)
