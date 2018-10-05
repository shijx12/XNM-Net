import torch
from tqdm import tqdm
import argparse
import numpy as np
import os
from IPython import embed

from DataLoader import ClevrDataLoader
from model.net import XNMNet
from utils.misc import todevice

def validate(model, data, device):
    count, correct = 0, 0
    model.eval()
    print('validate...')
    for batch in tqdm(data.generator(), total=len(data)):
        questions, answers, programs, program_inputs, conn_matrixes, cat_matrixes, pre_v = \
                [todevice(x, device) for x in batch]
        logits = model(programs, program_inputs, conn_matrixes, cat_matrixes, pre_v)
        predicts = logits.max(1)[1]
        correct += torch.eq(predicts, answers).long().sum().item()
        count += answers.size(0)
    acc = correct / count
    return acc


def show_edge_attention(model, data):
    print('*'*99)
    for word in data.vocab['edge_token_to_idx']:
        i = data.vocab['question_token_to_idx'][word]
        query = model.word_embedding.weight[i]
        attention = torch.matmul(query, model.edge_cat_vectors.t())
        print(' *** query: %s' % word)
        info = ''
        for i in range(len(data.vocab['edge_idx_to_token'])):
            token = data.vocab['edge_idx_to_token'][i]
            info += '%s: %.4f\t' % (token, attention[i].item())
        print(info)
    print('*'*99)
    print('\n'*3)


def visualize(model, data, device):
    model.eval()
    for batch in data.generator():
        questions, answers, programs, program_inputs, conn_matrixes, cat_matrixes, pre_v = \
                [todevice(x, device) for x in batch]
        predict_str, intermediates = model.forward_and_return_intermediates(programs, program_inputs, conn_matrixes, cat_matrixes, pre_v)
        answer_str = data.vocab['answer_idx_to_token'][answers[0].item()]
        if predict_str != answer_str:
            question_str = ' '.join(list(filter(lambda x: x!='<NULL>', [data.vocab['question_idx_to_token'][q.item()] for q in questions[0]])))
            program = [data.vocab['program_idx_to_token'][q.item()] for q in programs[0]]
            program_input = [data.vocab['question_idx_to_token'][q.item()] for q in program_inputs[0]]
            program_str = []
            for p, i in zip(program, program_input):
                if p not in {'<NULL>', '<START>', '<END>'}:
                    program_str.append(p+'_'+i if i!='<NULL>' else p)
            program_str = ' '.join(program_str)

            print("="*88)
            print("*"*33)
            desc = data.desc_cache[0]
            for _ in intermediates:
                if _ is None:
                    print(" >>>>>>>> new chain <<<<<<<<<<< ")
                    continue
                module_type, node_attn = _
                assert len(node_attn) == len(desc)
                print(" >> %s" % module_type)
                for weight, node_desc in zip(node_attn, desc):
                    print("%.3f\t%s" % (weight, node_desc))
                print(" <<<<<< ")
            print("question: %s" % question_str)
            print("answer: %s, predict: %s" % (answer_str, predict_str))
            print("program: %s" % program_str)

            embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input_dir', default='/data/sjx/CLEVR-Exp/data')
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--val_scene_pt', default='val_scenes.pt')
    parser.add_argument('--vocab_json', default='vocab.json')
    # model hyperparameters
    parser.add_argument('--dim_pre_v', default=15, type=int)
    parser.add_argument('--dim_v', default=128, type=int)
    # control parameters
    parser.add_argument('--mode', default='vis', choices=['vis', 'val'])
    args = parser.parse_args()

    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.val_question_pt)
    args.val_scene_pt = os.path.join(args.input_dir, args.val_scene_pt)
    
    device = 'cpu' if args.mode=='vis' else 'cuda'
    val_loader_kwargs = {
        'question_pt': args.val_question_pt,
        'scene_pt': args.val_scene_pt,
        'vocab_json': args.vocab_json,
        'batch_size': 1 if args.mode=='vis' else 128,
        'shuffle': False
    }
    val_loader = ClevrDataLoader(**val_loader_kwargs)
    model_kwargs = {
        'vocab': val_loader.vocab,
        'dim_pre_v': args.dim_pre_v,
        'dim_v': args.dim_v,
    }
    model = XNMNet(**model_kwargs).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})['state_dict'])

    if args.mode =='vis':
        visualize(model, val_loader, device)
    elif args.mode == 'val':
        val_acc = validate(model, val_loader, device)
        print("Validate acc: %.4f" % val_acc)
