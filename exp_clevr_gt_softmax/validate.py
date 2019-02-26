import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to import shared utils
import torch
from tqdm import tqdm
import argparse
import numpy as np
import os
import json
from IPython import embed

from DataLoader import ClevrDataLoader
from utils.generate_programs import generate_single_program, load_program_generator
from utils.misc import convert_david_program_to_mine, invert_dict, todevice
from model.net import XNMNet

map_program_to_cat = {
        'count': 'count',
        'equal': 'compare attribute',
        'equal_integer': 'compare number',
        'exist': 'exist',
        'greater_than': 'compare number',
        'less_than': 'compare number',
        'query': 'query',
        }


def validate(model, data, device, detail=False):
    count, correct = 0, 0
    model.eval()
    details = { cat:[0,0] for cat in {'count', 'compare number', 'exist', 'query', 'compare attribute'}}
    print('validate...')
    for batch in tqdm(data.generator(), total=len(data)):
        answers, questions, *batch_input = [todevice(x, device) for x in batch]
        logits, others = model(*batch_input)
        predicts = logits.max(1)[1]
        """
        There are some counting questions in CLEVR whose answer is a large number (such as 8 and 9). 
        However, as the training instances of such questions are very few, 
        the predictions of our softmax-based classifier can't reach a 100% accuracy for counting questions (we can only reach up to 99.99%). 
        Thanks to our attention mechanism over scene graphs, we can predict the answers of counting questions by directly summing up the node attention, 
        instead of feeding hidden features into a classifier. This alternative strategy gives a 100% counting accuracy.
        """
        # correct += torch.eq(predicts, answers).long().sum().item()
        count_outputs = others['count_outputs']
        for i in range(len(count_outputs)):
            if count_outputs[i] is None:
                correct += int(predicts[i].item()==answers[i].item())
            else:
                p = int(round(count_outputs[i].item()))
                a = int(data.vocab['answer_idx_to_token'][answers[i].item()])
                correct += int(p==a)
        count += answers.size(0)
        if detail:
            programs = batch_input[0]
            for i in range(len(answers)):
                for j in range(len(programs[i])):
                    program = data.vocab['program_idx_to_token'][programs[i][j].item()]
                    if program in ['<NULL>', '<START>', '<END>', '<UNK>', 'unique']:
                        continue
                    cat = map_program_to_cat[program]
                    if program == 'count':
                        p = int(round(count_outputs[i].item()))
                        a = int(data.vocab['answer_idx_to_token'][answers[i].item()])
                    else:
                        p = predicts[i].item()
                        a = answers[i].item()
                    details[cat][0] += int(p==a)
                    details[cat][1] += 1
                    break
    acc = correct / count
    if detail:
        details = { k:(v[0]/v[1]) for k,v in details.items() }
        return acc, details
    return acc


def validate_with_david_generated_program(model, data, device, pretrained_dir):
    program_generator = load_program_generator(os.path.join(pretrained_dir, 'program_generator.pt')).to(device)
    david_vocab = json.load(open(os.path.join(pretrained_dir, 'david_vocab.json')))
    david_vocab['program_idx_to_token'] = invert_dict(david_vocab['program_token_to_idx'])
    details = { cat:[0,0] for cat in {'count', 'compare number', 'exist', 'query', 'compare attribute'}}

    count, correct = 0, 0
    model.eval()
    print('validate...')
    for batch in tqdm(data.generator(), total=len(data)):
        answers, questions, gt_programs, gt_program_inputs, *batch_input = [todevice(x, device) for x in batch]
        programs, program_inputs = [], []
        # generate program using david model for each question
        for i in range(questions.size(0)):
            question_str = []
            for j in range(questions.size(1)):
                word = data.vocab['question_idx_to_token'][questions[i,j].item()]
                if word == '<START>': continue
                if word == '<END>': break
                question_str.append(word)
            question_str = ' '.join(question_str) # question string
            david_program = generate_single_program(question_str, program_generator, david_vocab, device)
            david_program = [david_vocab['program_idx_to_token'][i.item()] for i in david_program.squeeze()]
            # convert david program to ours. return two index lists
            program, program_input = convert_david_program_to_mine(david_program, data.vocab)
            programs.append(program)
            program_inputs.append(program_input)
        # padding
        max_len = max(len(p) for p in programs)
        for i in range(len(programs)):
            while len(programs[i]) < max_len:
                programs[i].append(vocab['program_token_to_idx']['<NULL>'])
                program_inputs[i].append(vocab['question_token_to_idx']['<NULL>'])
        # to tensor
        programs = torch.LongTensor(programs).to(device)
        program_inputs = torch.LongTensor(program_inputs).to(device)

        logits, others = model(programs, program_inputs, *batch_input)
        predicts = logits.max(1)[1]
        correct += torch.eq(predicts, answers).long().sum().item()
        count += answers.size(0)
        # details
        for i in range(len(answers)):
            for j in range(len(gt_programs[i])):
                program = data.vocab['program_idx_to_token'][gt_programs[i][j].item()]
                if program in ['<NULL>', '<START>', '<END>', '<UNK>', 'unique']:
                    continue
                cat = map_program_to_cat[program]
                details[cat][0] += int(predicts[i].item()==answers[i].item())
                details[cat][1] += 1
                break
    acc = correct / count
    details = { k:(v[0]/v[1]) for k,v in details.items() }
    return acc, details



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--val_question_pt', default='val_questions.pt')
    parser.add_argument('--val_scene_pt', default='val_scenes.pt')
    parser.add_argument('--vocab_json', default='vocab.json')
    parser.add_argument('--pretrained_dir', default='../pretrained')
    # control parameters
    parser.add_argument('--program', default='gt', choices=['gt', 'david'])
    args = parser.parse_args()

    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.val_question_pt)
    args.val_scene_pt = os.path.join(args.input_dir, args.val_scene_pt)
    
    device = 'cuda'
    val_loader_kwargs = {
        'question_pt': args.val_question_pt,
        'scene_pt': args.val_scene_pt,
        'vocab_json': args.vocab_json,
        'batch_size': 128,
        'shuffle': False
    }
    val_loader = ClevrDataLoader(**val_loader_kwargs)
    
    loaded = torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})
    model_kwargs = loaded['model_kwargs']
    model_kwargs.update({'vocab': val_loader.vocab})
    model = XNMNet(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'])

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(' ~~~~~~~~~~ num parameters: %d ~~~~~~~~~~~~~' % num_parameters)

    if args.program == 'gt':
        print('validate with **ground truth** program')
        val_acc, val_details = validate(model, val_loader, device, detail=True)
    elif args.program == 'david':
        print('validate with **david predicted** program')
        val_acc, val_details = validate_with_david_generated_program(model, val_loader, device, args.pretrained_dir)
    print("Validate acc: %.4f" % val_acc)
    print(json.dumps(val_details, indent=2))
