import torch
from tqdm import tqdm
import argparse
from utils.misc import todevice
import numpy as np
import os
import json
from utils.vqa import VQADataLoader
from model_stack.module_net import TbDNet
from IPython import embed

def validate(model, data, device):
    count, correct = 0, 0
    model.eval()
    print('validate...')
    for batch in tqdm(data.generator(), total=len(data)):
        questions, questions_len, gt_programs, answers, conn_matrixes, cat_matrixes, vertex_indexes = \
                [todevice(x, device) for x in batch]
        logits, others = model(questions, questions_len, conn_matrixes, cat_matrixes, vertex_indexes)
        predicts = logits.max(1)[1]
        correct += torch.eq(predicts, answers).long().sum().item()
        count += answers.size(0)
    acc = correct / count
    return acc


def visualize(model, data, device):
    model.eval()
    for batch in data.generator():
        questions, questions_len, gt_programs, answers, conn_matrixes, cat_matrixes, vertex_indexes = \
                [todevice(x, device) for x in batch]
        predict_str, intermediates = model.forward_and_return_intermediates(questions, questions_len, gt_programs, conn_matrixes, cat_matrixes, vertex_indexes)
        answer_str = data.vocab['answer_idx_to_token'][answers[0].item()]
        if True: #predict_str != answer_str:
            question_str = ' '.join(list(filter(lambda x: x!='<NULL>', [data.vocab['question_idx_to_token'][q.item()] for q in questions[0]])))

            print("="*88)
            print("*"*33)
            desc = data.desc_cache[0]
            for module_type, node_attn in intermediates:
                print(" >> %s" % module_type)
                if node_attn is None:
                    continue
                assert len(node_attn) == len(desc)
                for weight, node_desc in zip(node_attn, desc):
                    print("%.3f\t%s" % (weight, node_desc))
                print(" <<<<<< ")
            print("question: %s" % question_str)
            print("answer: %s, predict: %s" % (answer_str, predict_str))
            print("program: %s" % program_str)

            embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='val', choices=['val', 'vis'])
    parser.add_argument('--dropout', default=0.5, type=float)
    # input
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input_dir', default='/data/sjx/VQA-Exp/data')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--val_scene_pt', default='val_scenes.pt')
    parser.add_argument('--vocab_json', default='vocab.json')
    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int, help='word embedding')
    parser.add_argument('--dim_hidden', default=300, type=int, help='hidden state of seq2seq parser')
    parser.add_argument('--dim_v', default=300, type=int, help='node/attribute/edge embedding')
    args = parser.parse_args()

    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.val_question_pt)
    args.val_scene_pt = os.path.join(args.input_dir, args.val_scene_pt)
    
    device = 'cuda' if args.mode == 'val' else 'cpu'
    val_loader_kwargs = {
        'question_pt': args.val_question_pt,
        'scene_pt': args.val_scene_pt,
        'vocab_json': args.vocab_json,
        'batch_size': 128 if args.mode == 'val' else 1,
        'shuffle': False
    }
    val_loader = VQADataLoader(**val_loader_kwargs)
    model_kwargs = {
        'vocab': val_loader.vocab,
        'dim_v': args.dim_v,
        'dim_word': args.dim_word,
        'dim_hidden': args.dim_hidden,
        'dropout': args.dropout,
        'max_decoder_len': val_loader.max_program_len,
        'device': device,
    }
    model = TbDNet(**model_kwargs).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})['state_dict'])

    if args.mode == 'val':
        valid_acc = validate(model, val_loader, device)
    else:
        visualize(model, val_loader, device)
    # cmd: python3.6 validate.py --ckpt /data/sjx/CLEVR-Exp/acc100/model.pt


