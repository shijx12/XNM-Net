#!/usr/bin/env python3
import re
import os
import argparse
import json
import numpy as np
import pickle
from utils import encode
from collections import Counter


"""
Preprocessing script for VQA question files.
"""

# according to https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/data.py
_special_chars = re.compile('[^a-z0-9 ]*')
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def process_punctuation(s):
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    return s.strip()



def main(args):
    print('Loading data')
    annotations, questions = [], []
    if args.input_annotations_json is not None:
        for f in args.input_annotations_json.split(':'):
            annotations += json.load(open(f, 'r'))['annotations']
    for f in args.input_questions_json.split(':'):
        questions += json.load(open(f, 'r'))['questions']
    print('number of questions: %s' % len(questions))
    question_id_to_str = { q['question_id']:q['question'] for q in questions }
    if args.mode != 'test':
        assert len(annotations) > 0

    # Either create the vocab or load it from disk
    if args.mode == 'train':
        print('Building vocab')
        answer_cnt = {}
        for ann in annotations:
            answers = [_['answer'] for _ in ann['answers']]
            for i,answer in enumerate(answers):
                answer = process_punctuation(answer)
                answer_cnt[answer] = answer_cnt.get(answer, 0) + 1
                answers[i] = answer
            ann['answers'] = answers # update
        answer_token_to_idx = {}
        for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
            answer_token_to_idx[token] = len(answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>':0, '<UNK>':1}
        for i,q in question_id_to_str.items():
            question = q.lower()[:-1]
            question = _special_chars.sub('', question)
            question_id_to_str[i] = question
            for token in question.split(' '):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'program_token_to_idx': {token:i for i,token in enumerate(['<eos>','find','relate','describe','is','and'])}
        }
        
        print('Write into %s' % args.vocab_json)
        with open(args.vocab_json, 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        print('Loading vocab')
        with open(args.vocab_json, 'r') as f:
            vocab = json.load(f)
        for ann in annotations:
            answers = [_['answer'] for _ in ann['answers']]
            for i,answer in enumerate(answers):
                answer = process_punctuation(answer)
                answers[i] = answer
            ann['answers'] = answers # update
        for i,q in question_id_to_str.items():
            question = q.lower()[:-1]
            question = _special_chars.sub('', question)
            question_id_to_str[i] = question


    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    image_idxs = []
    answers = []
    if args.mode in {'train', 'val'}:
        for a in annotations:
            question = question_id_to_str[a['question_id']]
            question_tokens = question.split(' ')
            question_encoded = encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            image_idxs.append(a['image_id'])

            answer = [] 
            for per_ans in a['answers']:
                if per_ans in vocab['answer_token_to_idx']:
                    i = vocab['answer_token_to_idx'][per_ans]
                    answer.append(i)
            answers.append(answer)
    elif args.mode == 'test':
        for q in questions: # remain the original order to match the question_id
            question = question_id_to_str[q['question_id']] # processed question
            question_tokens = question.split(' ')
            question_encoded = encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            image_idxs.append(q['image_id'])
            answers.append([0])

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    glove_matrix = None
    if args.mode == 'train':
        token_itow = { i:w for w,i in vocab['question_token_to_idx'].items() }
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing')
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'image_idxs': np.asarray(image_idxs),
        'answers': answers,
        'glove': glove_matrix,
        }
    with open(args.output_pt, 'wb') as f:
        pickle.dump(obj, f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_top', default=3000, type=int)
    parser.add_argument('--glove_pt', help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--input_questions_json', required=True)
    parser.add_argument('--input_annotations_json', help='not need for test mode')
    parser.add_argument('--output_pt', required=True)
    parser.add_argument('--vocab_json', required=True)
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    main(args)
