import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

'''
train_data = batchify(corpus.train, args.batch_size)
nbatch = data.size(0) // nbatch = 데이터의 총길이
data = data.narrow(0, 0, nbatch * bsz) // Tensor.narrow(dimension, start, length) → Tensor
-- dimension (int) – 어떤 디멘션을 줄일지.
-- start (int) – 그 디멘션의 몇번째부터 쓸지.
-- length (int) – 총 길이.
---> 한줄로 쭉 폈다고 볼 수 있음. (이미 펴져있음)
'''
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    
    # Evenly divide the data across the bsz batches.
    
    data = data.view(bsz, -1).t().contiguous()
    '''
    data = data.view(bsz, -1).t().contiguous()
    data.view(bsz, -1) // [929580]  -> [10,92958]  그리고 transpose (bsz =10 두었을때)
    결과적으로 [92958,10] 배치 사이즈만큼씩 가져오기 쉽게됨.
    language model이고 val_data와 test_data 모두 같은방식으로하니까 순서가 어긋나진 않음.
    
    ''' 
    #Tensor.contiguous() -> 메모리에 가능한 인접하게 둔다는 명령인듯.(추측)
    
    if args.cuda:
        data = data.cuda()
       
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

#dictionay에는 단어 들어있음.
ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

#리패키지를 하는이유 == Truncated backpropagation through time (BPTT) 의 기법으로.
#backprop 이 매번 sequence 길이 안에서만 진행되도록. -> 그래디언트가 앞까지 이어진다면 매번 계산량이 많기에, Variable의 그래프를 초기화.

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

#bptt length(35) 만큼 가져오게됨. 이걸 겟 배치라고 하는데, args.batch_size(20) 도 배치라고 부르는데 편한대로 부르는듯.
#[35,20]에서 뒤의 20은 hidden의 유닛수와 같은 배치사이즈[2,20,200] ( init_hidden() ) 꼭 같을 필요 없는지는 확인필요. -> 같아야함.
# 그러므로 35의 배치는 데이터에서 가져오는 단위이고, 20의 배치는 RNN에 한번에 같이들어가는 크기라고 볼 수 있나.?
# nn.RNN() 에 들어갈때는 [35,20,200(emb_size)] 로 다같이 들어감.

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        #data size = [35,10]
        output, hidden = model(data, hidden)
        
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        #여기서도 repackage를 하는 이유는 알 수 없다. 학습과정이 아닌데도.
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


'''
### trainnig data size 변화과정 ###

Data.data[929584] ---Main.batchfy()---> [46479,20(batch_size)] ---Main.get_batch()--->  [35(bptt len),20]
---Model.encoder(embedding)---> [35,20,200(embedding_size)] ---Model.nn.RNN---> [35,20,200] ---Model.transform---> [700,200] 
---Model.decoder(LinearLayer[200,10000]---> [700,10000] ---Model.ret_view---> [35,20,10000] == output. (10000개 단어에 대한 확률 값)

'''
def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    #임의 히든스테이트 만듬. 거의 무의미한 값으로 예상(Model 에서 확인).
    hidden = model.init_hidden(args.batch_size)
    
    #tra data length = 46479, batch_len=35 --> 1327 batches. 35 배치사이즈만큼씩 20(배치사이즈)개 진행.
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        # output.view(-1, ntokens)의 size는 [700,10000]고 targets는 [700]. 두 사이즈가 이렇게 다른 이유는 target에는 0부터 10000까지의 값이 들어있기 때문.
        # output view에는 700개의 단어의 10000차원의 확률이 들어있고. target은 그중에서 몇번째 인덱스의 값이 답인지 알려줌.
        # target 이 [700,10000] 의 원핫벡터일 수도 있겠지만 공간낭비.
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        #optimizer 없이 직접 업데이트 해주는 방법.
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)
        #loss를 모으다가 log_interval마다 정보 출력.
        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        
        #training 데이터를 x epoch 번 만큼 학습.
        train()
        
        #validation loss.
        val_loss = evaluate(val_data)
        
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        # 트레이닝중에 가장 좋았던 model을 save. byte로. -- 모델을 파일로 저장하는 방법. torch.save(model,f)
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# 가장 학습잘된 모델로 마지막 테스트 data 사용하여 평가.
# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)