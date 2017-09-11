import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    # idx2word 에 단어 추가, word2idx 는 word 가 key, 이에대한 idx 가 value 값을 가짐.
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    
    
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            # line은 <eos> end of sentence 단위 하나의 문장을 token으로 정의. 문장 안의 단어는 class Dictionary의 .add_word()로 추가
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        '''
        ids = torch.LongTensor(tokens) - 파일안의 단어 총 갯수 만큼 long tensor 할당 (배열처럼) ids 는 롱텐서배열. 
        파일 안에있는 모든 라인에 대해 word 단위로 그 단어 들의 dict로 만든 인덱스 값이 전부 들어감.
        따라서 idx[] 안에는 모든 단어들이 출현 순서로 정해진 index 값으로 들어가 있음. 각 단어의 index 값은 0부터 출현 순서에따라 임의 배정
        이값들이 corpus.train, test, valid 값임.
        *torch.LongTensor() : 64bit signed 인트 텐서
    
        '''
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids