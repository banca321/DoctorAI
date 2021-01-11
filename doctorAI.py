import numpy as np
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim


def load_data(seqFile, labelFile):
    train_set_x = pickle.load(open(seqFile+'.train', 'rb'))
    valid_set_x = pickle.load(open(seqFile+'.valid', 'rb'))
    test_set_x = pickle.load(open(seqFile+'.test', 'rb'))
    train_set_y = pickle.load(open(labelFile+'.train', 'rb'))
    valid_set_y = pickle.load(open(labelFile+'.valid', 'rb'))
    test_set_y = pickle.load(open(labelFile+'.test', 'rb'))
    
    #print('train_set: ', len(train_set_y))
    #print('valid_set: ', len(valid_set_y))
    #print('test_set: ', len(test_set_y))
    
    def len_argsort(seq):    #sort the datasetto patient's with least to most admissions
        return sorted(range(len(seq)), key=lambda x: len(seq[x]), reverse=True)
    
    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]
    
    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]
    
    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]
    
    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)
    
    return train_set, valid_set, test_set
    
def padMatrix(seqs, labels, input_size, num_classes):
    lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    maxlen = np.max(lengths)
    
    #print('maxlen = ', maxlen)
    
    x = torch.zeros([maxlen, n_samples, input_size], dtype=torch.int64)
    y = torch.zeros([maxlen, n_samples, num_classes], dtype=torch.int64)
    
    for idx, (seq,label) in enumerate(zip(seqs,labels)):
        for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
            xvec[subseq] = 1
        for yvec, subseq in zip(y[:,idx,:], label[1:]):
            yvec[subseq] = 1
    
    lengths = torch.from_numpy(lengths)
    return x, y, lengths

def calculate_auc(model, dataset, batchSize, input_size, num_classes, criterion):
    
    n_batches = int(np.ceil(float(len(dataset[0]))/float(batchSize)))
    aucSum = 0.0
    dataCount = 0.0
    
    for index in range(n_batches):
        batchX = dataset[0][index*batchSize:(index+1)*batchSize]
        batchY = dataset[1][index*batchSize:(index+1)*batchSize]
        size = len(batchX)
        
        batchX, batchY, lengths = padMatrix(batchX, batchY, input_size, num_classes)
        
                    
        
        #print('size: ', size)
        h = model.init_hidden(size)

        auc, h = model(batchX.float(), lengths, h)
        batchY = batchY.to(dtype= torch.float)

        auc = criterion(auc, batchY)
        
        aucSum += auc * len(batchX)
        dataCount += float(len(batchX))
        
    return aucSum/dataCount


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes, embed_size= 200, n_layers=2, drop_prob=0.5):
        
        super(GRUNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers        
        self.num_classes = num_classes   
        self.embed_size = embed_size
        
        #self.emb = nn.Embedding(input_size, embed_size)
        
        
        self.emb = nn.Linear(input_size, embed_size)
        
        
        self.gru = nn.GRU(embed_size, hidden_dim, n_layers, bias=True, dropout=drop_prob)
        
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_dim, num_classes)  
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x, lengths, h):
    

        length = x.size(0)
        batch_size = x.size(1)
        input_size = x.size(2)
        
        x = x.view(-1, self.input_size)
        #print('x1: ', x.size())
        #b_size = x.size(0)
        x = self.emb(x)
        x = x.view(-1, batch_size, self.embed_size)
        #print('x2: ', x.size())
        
        
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False)
        #print('h: ', h.shape)
        
        out, h = self.gru(packed, h)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(out)
        #out = self.dropout(out)
        out = self.linear(out)
        out = self.sigmoid(out)
        
        return out, h
    
    def init_weight(self):
        
        self.emb.weight = torch.nn.init.uniform_(self.emb.weight, a=-0.01, b=0.01)
        self.emb.bias = torch.nn.init.zeros_(self.emb.bias)
        
        self.gru.weight_ih_l0 = torch.nn.init.uniform_(self.gru.weight_ih_l0, a=-0.01, b=0.01)
        self.gru.weight_hh_l0  = torch.nn.init.uniform_(self.gru.weight_hh_l0, a=-0.01, b=0.01)
        self.gru.bias_ih_l0 = torch.nn.init.zeros_(self.gru.bias_ih_l0)
        self.gru.bias_hh_l0 = torch.nn.init.zeros_(self.gru.bias_hh_l0)
        self.gru.weight_ih_l1 = torch.nn.init.uniform_(self.gru.weight_ih_l1, a=-0.01, b=0.01)
        self.gru.weight_hh_l1 = torch.nn.init.uniform_(self.gru.weight_hh_l1, a=-0.01, b=0.01)
        self.gru.bias_ih_l1 = torch.nn.init.zeros_(self.gru.bias_ih_l1)
        self.gru.bias_hh_l1 = torch.nn.init.zeros_(self.gru.bias_hh_l1)

        self.linear.weight = torch.nn.init.uniform_(self.linear.weight, a=-0.01, b=0.01)
        self.linear.bias = torch.nn.init.zeros_(self.linear.bias)
        
        
        '''
        
        self.emb.weight.data.fill_(0.01)
        self.emb.bias.data.fill_(0.01)
        
        self.gru.weight_ih_l0.data.fill_(0.01)
        self.gru.weight_hh_l0.data.fill_(0.01)
        self.gru.bias_ih_l0.data.fill_(0.01)
        self.gru.bias_hh_l0.data.fill_(0.01)
        self.gru.weight_ih_l1.data.fill_(0.01)
        self.gru.weight_hh_l1.data.fill_(0.01)
        self.gru.bias_ih_l1.data.fill_(0.01)
        self.gru.bias_hh_l1.data.fill_(0.01)

        self.linear.weight.data.fill_(0.01)
        self.linear.bias.data.fill_(0.01)
        '''
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden
        
       
def train_doctorAI(seqFile, labelFile, input_size, hidden_dim, num_classes, embed_size, batchSize=100, max_epochs=20, dropout_rate=0.5):
    
    
    print("Loading Data ... ")
    trainSet, validSet, testSet = load_data(seqFile, labelFile)
    n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
    train_size = len(trainSet[0])
    
    print("Done")
    
    

    model = GRUNet(input_size, hidden_dim, num_classes, embed_size)
    model.init_weight()
   
    criterion = nn.BCELoss()
    #learning_rate = 1.0
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)   #need to know the learning rate
   
    
    bestValidCrossEntropy = 1e20
    bestValidEpoch = 0
    testCrossEntropy = 0.0
    
    print("Optimization Start!!")
    
    
    for epoch in range(max_epochs):
        model.train()
        iteration = 0
        running_loss = 0
        costVector = []
        
        #h = model.init_hidden(batchSize)
        
        #for index in range(n_batches-1):
        for index in random.sample(range(n_batches), n_batches):
            
            batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
            batchY = trainSet[1][index*batchSize:(index+1)*batchSize]       
            
            size = len(batchX)
            
            batchX, batchY, lengths = padMatrix(batchX, batchY, input_size, num_classes)
            
            optimizer.zero_grad()
            
            h = model.init_hidden(size)
            
            out, h = model(batchX.float(), lengths, h)
            
            #print("shape of out", out.shape)
         
            batchY = batchY.to(dtype= torch.float)
           
            #print('shape of out: ', out.shape)
            #print('shape of batchY: ', batchY.shape)
            
            cost = criterion(out, batchY)
            cost.backward()
            #for param in model.parameters():
                #print(param.grad.data)
                
            #print('linear weight: ', model.linear.weight.grad)
            #print('linear bias: ', model.linear.bias.grad)
        
            #print('gru first layer weight ih: ', model.gru.weight_ih_l0.grad)
            #print('gru first layer weight hh: ',model.gru.weight_hh_l0.grad)
            #print('gru first layer bias ih: ',model.gru.bias_ih_l0.grad)
            #print('gru first layer bias hh: ',model.gru.bias_hh_l0.grad)
            optimizer.step()
            
            running_loss += cost.item()
          
        
        else:
            
            with torch.no_grad():
                model.eval()
                
                #print("Model's state_dict:")
                #for param_tensor in model.state_dict():
                    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                
                validAuc = calculate_auc(model, validSet, batchSize, input_size, num_classes, criterion)
                print("Validation cross entropy:%f at epoch:%d" % (validAuc, epoch))
                if validAuc < bestValidCrossEntropy:
                    bestValidCrossEntropy = validAuc
                    bestValidEpoch = epoch
                    
                    testCrossEntropy = calculate_auc(model, testSet, batchSize, input_size, num_classes, criterion)
                    
                    print('Test Cross Entropy:%f at Epoch:%d' % (testCrossEntropy, epoch))
                    #print(list(model.named_parameters()))
                    
                    torch.save(model, 'model2.'+str(epoch)+'.pth')
                   
        print("Epoch:%d, Mean_Cost:%f" % (epoch, running_loss/n_batches))
                    
                
            
            
    print('The Best Valid Cross Entropy:%f at epoch:%d' % (bestValidCrossEntropy, bestValidEpoch))
    print('The Test Cross Entropy: %f' % testCrossEntropy)
    
    
    
