import numpy as np
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab


def collate(batch):
    batch = list(zip(*batch))
    answer, question, program_seq, program_input = list(map(torch.stack, batch[:4]))
    return (answer, question, program_seq, program_input, *batch[4:])


class ClevrDataset(Dataset):

    def __init__(self, questions, image_indices, programs, program_inputs, answers, features, idx_cache, coord_cache):
        # convert data to tensor
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_image_idxs = torch.LongTensor(np.asarray(image_indices))
        self.all_programs = torch.LongTensor(np.asarray(programs))
        self.all_program_inputs = torch.LongTensor(np.asarray(program_inputs))
        self.all_answers = torch.LongTensor(np.asarray(answers))
        self.features = features
        self.idx_cache = idx_cache
        self.coord_cache = coord_cache

    def __getitem__(self, index):
        question = self.all_questions[index]
        image_idx = self.all_image_idxs[index].item()
        program_seq = self.all_programs[index]
        program_input = self.all_program_inputs[index]
        answer = self.all_answers[index]
        assert program_seq.size(0) == program_input.size(0), "program and program_input must have the same length"

        feature, coord = self.features[image_idx]['feature'], self.features[image_idx]['coord'][:,:2]
        feature = torch.FloatTensor(feature)
        num_obj = len(coord)
        edge_vector = np.zeros((num_obj, num_obj, 2))
        for i in range(num_obj):
            for j in range(num_obj):
                edge_vector[i,j] = coord[i] - coord[j]
        edge_vector = torch.FloatTensor(edge_vector)

        self.idx_cache[0] = image_idx
        self.coord_cache[0] = coord
        return (answer, question, program_seq, program_input, feature, edge_vector)

    def __len__(self):
        return len(self.all_questions)



class ClevrDataLoader(DataLoader):

    def __init__(self, **kwargs):
        if 'question_pt' not in kwargs:
            raise ValueError('Must give question_pt')
        if 'feature_pt' not in kwargs:
            raise ValueError('Must give feature_pt')
        if 'vocab_json' not in kwargs:
            raise ValueError('Must give vocab_json')

        feature_pt_path = str(kwargs.pop('feature_pt'))
        with open(feature_pt_path, 'rb') as f:
            features = pickle.load(f)

        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            image_indices = obj['image_idxs']
            programs = obj['programs']
            program_inputs = obj['program_inputs']
            answers = obj['answers']

        self.ratio = None
        if 'ratio' in kwargs:
            self.ratio = kwargs.pop('ratio')
            total = int(len(questions) * self.ratio)
            print('training ratio = %.3f, containing %d questions' % (self.ratio, total))
            questions = questions[:total]
            image_indices = image_indices[:total]
            programs = programs[:total]
            program_inputs = program_inputs[:total]
            answers = answers[:total]

        self.idx_cache, self.coord_cache = [0], [0] # just store some information for visualization
        dataset = ClevrDataset(questions, image_indices, programs, program_inputs, answers, features, 
                               self.idx_cache, self.coord_cache)
        kwargs['collate_fn'] = collate   
        super().__init__(dataset, **kwargs)
        self.vocab = vocab

