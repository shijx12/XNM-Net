# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from IPython import embed


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
    return vocab


class VQADataset(Dataset):

    def __init__(self, answers, questions, questions_len, q_image_indices,
                       feature_h5, feat_coco_id_to_index, num_answer):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_q_image_idxs = torch.LongTensor(np.asarray(q_image_indices))

        self.feature_h5 = feature_h5
        self.feat_coco_id_to_index = feat_coco_id_to_index
        self.num_answer = num_answer


    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        if answer is not None:
            _answer = torch.zeros(self.num_answer)
            for i in answer:
                _answer[i] += 1
            answer = _answer
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        image_idx = self.all_q_image_idxs[index].item() # coco_id
        # fetch vision features
        vision_feat = self._load_image(image_idx)

        return (image_idx, answer, question, question_len, vision_feat)

    def __len__(self):
        return len(self.all_questions)

    def _load_image(self, image_id):
        """ Load an image """
        index = self.feat_coco_id_to_index[image_id]
        vision_feat = self.feature_h5[index]
        return torch.from_numpy(vision_feat)


class VQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            q_image_indices = obj['image_idxs']
            answers = obj['answers']
            glove_matrix = obj['glove']
        
        with h5py.File(kwargs['feature_h5'], 'r') as features_file:
            coco_ids = features_file['ids'][()]
        feat_coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        self.feature_h5 = h5py.File(kwargs.pop('feature_h5'), 'r')['features']

        self.dataset = VQADataset(answers, questions, questions_len, q_image_indices,
                self.feature_h5, feat_coco_id_to_index, len(vocab['answer_token_to_idx']))
        
        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        kwargs['collate_fn'] = default_collate
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def close(self):
        if self.feature_h5 is not None:
            self.feature_h5.close()
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


