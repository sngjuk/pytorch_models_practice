###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

#temperature 값으로 diversity를 조정한다고함.
if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")
#모델 불러오는 방법. torch.load(f), open(f,rb)
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

#model.cuda()
if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
#hidden size = [2,1,200]
# 1은 batch size

input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
#input size[1,1] 0부터 1까지의 수 x ntokens(단어의 수) 10000. volatile=true <- 이 값에대해선 backward() 계산 x 그래드 필요없음.

if args.cuda:
    input.data = input.data.cuda()

with open(args.outf, 'w') as outf:
    for i in range(args.words):
        #model([1,1],[2,1,200])
        output, hidden = model(input, hidden)
        #output size = [1,1,10000]
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        #output.squeeze() -> [10000], 더큰 temperature 값으로 나눌수록 값들간 차이가 작아지는것?.exp(), cpu()- cpu 텐서 반환.
        # .exp()를 하는 이유는 알수 없는데 아마도 multinomial을 위해 양수로 만든것 같음.
        
        word_idx = torch.multinomial(word_weights, 1)[0]
        #///////////////////////////의문///////////////////////////////////////
        # word_idx로 단어의 인덱스를 얻게됨. multinomial이 어떻게 작동하는지 모르겠음.
        # 가장 확률이 큰 값을 뽑아낼거라고 생각하지만, multinomial이 어떤역할인지 모름.
        # 배열로 실험해도 multinomial이 항상 가장 큰 값의 index를 뽑아내는것이 아님.
        #////////////////////////////////////////////////////////////////////////
        
        input.data.fill_(word_idx)
        #input의 데이터를 word_idx값으로 교체 (다음 단어 생성을 위해)
        word = corpus.dictionary.idx2word[word_idx]
        
        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))