#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse

import json
import os

import numpy as np
import pickle

import programs
from utils import tokenize, encode, build_vocab


"""
Preprocessing script for CLEVR question files.
"""


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='prefix',
                    choices=['chain', 'prefix', 'postfix'])
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--unk_threshold', default=1, type=int)
parser.add_argument('--encode_unk', default=0, type=int)

parser.add_argument('--input_questions_json', required=True)
parser.add_argument('--output_pt_file', required=True)
parser.add_argument('--output_vocab_json', default='')


def program_to_strs(program, mode):
  if mode == 'chain':
    if not programs.is_chain(program):
      return None
  elif mode == 'prefix':
    program = programs.list_to_prefix(program)
  elif mode == 'postfix':
    program = programs.list_to_postfix(program)

  ######################### convert program and program_inputs ##########################
  for f in program:
    if f['function'] in {'equal_shape', 'equal_color', 'equal_size', 'equal_material'}:
      f['function'] = 'equal'
    elif 'query' in f['function']:
      value = f['function'][6:] # <cat> of query_<cat>
      f['function'] = 'query'
      f['value_inputs'].append(value)
      assert len(f['value_inputs']) == 1
    elif 'same' in f['function']:
      value = f['function'][5:] # <cat> of same_<cat>
      f['function'] = 'same'
      f['value_inputs'].append(value)
    elif 'filter_' in f['function']:
      f['function'] = 'filter'
    
    if len(f['value_inputs']) == 0:
        f['value_inputs'].append('<NULL>')
    assert len(f['value_inputs']) == 1
  ####################################################################################

  func_str = ' '.join(f['function'] for f in program)
  input_str = ' '.join(f['value_inputs'][0] for f in program)
  return func_str, input_str


def main(args):
  print('Loading data')
  with open(args.input_questions_json, 'r') as f:
    questions = json.load(f)['questions']

  # Either create the vocab or load it from disk
  if args.input_vocab_json == '' or args.expand_vocab == 1:
    print('Building vocab')
    if 'answer' in questions[0]:
      answer_token_to_idx = build_vocab(
        (q['answer'] for q in questions)
      )
    question_token_to_idx = build_vocab(
      (q['question'] for q in questions),
      min_token_count=args.unk_threshold,
      punct_to_keep=[';', ','], punct_to_remove=['?', '.'],
      add_special=True
    )
    all_program_strs = []
    for q in questions:
      if 'program' not in q: continue
      program_str = program_to_strs(q['program'], args.mode)[0]
      if program_str is not None:
        all_program_strs.append(program_str)
    program_token_to_idx = build_vocab(all_program_strs, add_special=True)
    vocab = {
      'question_token_to_idx': question_token_to_idx,
      'program_token_to_idx': program_token_to_idx,
      'answer_token_to_idx': answer_token_to_idx, # no special tokens
    }

  if args.input_vocab_json != '':
    print('Loading vocab')
    if args.expand_vocab == 1:
      new_vocab = vocab
    with open(args.input_vocab_json, 'r') as f:
      vocab = json.load(f)
    if args.expand_vocab == 1:
      num_new_words = 0
      for word in new_vocab['question_token_to_idx']:
        if word not in vocab['question_token_to_idx']:
          print('Found new word %s' % word)
          idx = len(vocab['question_token_to_idx'])
          vocab['question_token_to_idx'][word] = idx
          num_new_words += 1
      print('Found %d new words' % num_new_words)

  if args.output_vocab_json != '':
    with open(args.output_vocab_json, 'w') as f:
      json.dump(vocab, f, indent=4)

  # Encode all questions and programs
  print('Encoding data')
  questions_encoded = []
  programs_encoded = []
  # value_inputs, encoded by question_token_to_idx in CLEVR
  # because all valid inputs are in question vocab
  program_inputs_encoded = [] 
  question_families = []
  orig_idxs = []
  image_idxs = []
  answers = []
  for orig_idx, q in enumerate(questions):
    question = q['question']

    orig_idxs.append(orig_idx)
    image_idxs.append(q['image_index'])
    if 'question_family_index' in q:
      question_families.append(q['question_family_index'])
    question_tokens = tokenize(question,
                        punct_to_keep=[';', ','],
                        punct_to_remove=['?', '.'])
    question_encoded = encode(question_tokens,
                         vocab['question_token_to_idx'],
                         allow_unk=args.encode_unk == 1)
    questions_encoded.append(question_encoded)

    if 'program' in q:
      program = q['program']
      program_str, input_str = program_to_strs(program, args.mode)
      program_tokens = tokenize(program_str)
      program_encoded = encode(program_tokens, vocab['program_token_to_idx'])
      programs_encoded.append(program_encoded)
      # program value_inputs
      input_tokens = tokenize(input_str)
      input_encoded = encode(input_tokens, vocab['question_token_to_idx'])
      assert len(input_encoded) == len(program_encoded) # input should have the same len with func
      program_inputs_encoded.append(input_encoded)

    if 'answer' in q:
      answers.append(vocab['answer_token_to_idx'][q['answer']])

  # Pad encoded questions and programs
  max_question_length = max(len(x) for x in questions_encoded)
  for qe in questions_encoded:
    while len(qe) < max_question_length:
      qe.append(vocab['question_token_to_idx']['<NULL>'])

  if len(programs_encoded) > 0:
    max_program_length = max(len(x) for x in programs_encoded)
    for pe in programs_encoded:
      while len(pe) < max_program_length:
        pe.append(vocab['program_token_to_idx']['<NULL>'])
    for ie in program_inputs_encoded:
      while len(ie) < max_program_length:
        ie.append(vocab['question_token_to_idx']['<NULL>'])

  questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
  programs_encoded = np.asarray(programs_encoded, dtype=np.int32)
  program_inputs_encoded = np.asarray(program_inputs_encoded, dtype=np.int32)
  print(questions_encoded.shape)
  print(programs_encoded.shape)
  print(program_inputs_encoded.shape)
  print('Writing')
  obj = {
          'questions': questions_encoded,
          'image_idxs': np.asarray(image_idxs),
          'orig_idxs': np.asarray(orig_idxs),
          'programs': programs_encoded,
          'program_inputs': program_inputs_encoded,
          'question_families': question_families,
          'answers': answers,
          }
  with open(args.output_pt_file, 'wb') as f:
    pickle.dump(obj, f)
 



if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
