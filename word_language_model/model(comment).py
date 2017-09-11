import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        
        #Dropout : 트레이닝 과정중 랜덤하게 input tensor의 몇몇 값을 0으로 만듦 p의 확률로.베르누이 distribut을 통해 샘플.
        #매번 fowrad 콜마다 랜덤하게. -- co-adaptation of neurons 에 효과적 ( 옆에있는뉴런들이 비슷하게 학습하는걸 막는걸로 이해됨)
        self.drop = nn.Dropout(dropout)
        
        #ntoken <-워드딕셔너리길이 ninp <- emsize 임베딩사이즈 (default =200), 벡터의 크기 결정 --차원수 줄이기위해서. 인코딩하는듯.
        #parameter는 대상 딕셔너리의 길이와 만들어낼 벡터 크기 (테이블이라고함). 클래스, variable =weights
        self.encoder = nn.Embedding(ntoken, ninp)
        
        #nn.LSTM 또는 GRU를 만들기위해서.
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            
        #디코딩 레이어-- nhid =히든레이어당 유닛수 기본200, ntoken=word dict의 길이 10000
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # encoding과 decoding weight를 같은걸 사용하는 옵션. 이경우 nhid와 emb 사이즈가 같아야한다고.
        #현재 nhid = 200, emb도 200.
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    # -0.1 ~ 0.1로 초기화
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    #emb [35,20,200]과 hidden [2,20,200] 이 nn.RNN 안에서 어떻게 조합되는것? 
    #2는 레이어의 갯수, 10 배치사이즈, 200임베딩 사이즈. 35는 backpropagtion through time로 bp 하는 단위 길이.
    #nn.RNN을 사용할때 input 모양과 같은 모양으로 hidden을 정의하고, 그냥넣음.
    #LinearLayer를 사용하는 경우엔 input의 컬럼길이와 layer의 row 길이가 같지만, (input [x,y] layer [y,z])
    # RNN의 경우는 input과 hidden의 모양이 같다. (LSTM은 히든 한개더 필요.)
    
    def forward(self, input, hidden):
        #input : [35(batch_size),20(tra_btch_sze)]
        #여기서 batch 사이즈 만큼 매번 인풋이 들어오는데 이에대해 매번 새로 embedding을 한다. 
        #임베딩이 모든 단어를 보고하는게아니고, 배치만큼만 보고 할 수 있는 이유는..?
        # 매 배치마다 같은 단어임에도 불구하고, 함께 들어온 다른 단어들에 영향을 받아 다른 벡터의 값을 갖게 될 수 있는 것 아닌지?
        # ---> 아님. Tensorflow의 word2vec과 달리 여기서는 그냥 weight들로 만들어내기만함. backprop은 bptt가 끝날 때마다.
        # 메인에 있는 model.parameters() 업데이트로 이루어 질것임.
        
        
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        #weight.size() = [10000,200] 10000x200 인 이유는 모르지만. 특별한 이유가 없을 수 도 있다. (밑에서 다짤림.)
        #weight type = FloatTensor
        
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            #weight는 [10000,200]이지만 weight.new()로 만들어진 이 Variable은 [2,20,200]이다. 
            #200만개가 8000개의 size로 변하는데, 이런식으로 사용하면 뒤의 값들은 모두 잘림.
            #initial state이기 때문에 특별히 의미있는 값이 가정되지 않는다고 생각함.그저 size 맞추기. lstm의 경우 [2,2,20,200]
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
