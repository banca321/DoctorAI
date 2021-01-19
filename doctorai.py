import numpy as np
import pickle
import random
import argparse

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
        h = model.init_hidden(size)

        auc, h = model(batchX.float(), lengths, h)
        batchY = batchY.to(dtype= torch.float)

        auc = criterion(auc, batchY)
        
        aucSum += auc * len(batchX)
        dataCount += float(len(batchX))
        
    return aucSum/dataCount



class GRUNet(nn.Module):
    def __init__(self, args):
        super(GRUNet, self).__init__()
        self.input_size = args.input_size
        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers        
        self.num_classes = args.num_classes   
        self.embed_size = args.embed_size
        
        #self.emb = nn.Embedding(input_size, embed_size)
        self.emb = nn.Linear(args.input_size, args.embed_size)
        
        self.gru = nn.GRU(args.embed_size, args.hidden_dim, args.n_layers, bias=True, dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.hidden_dim, args.num_classes)  
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x, lengths, h):
        batch_size = x.size(1)
        
        x = x.view(-1, self.input_size)
        x = self.emb(x)
        x = x.view(-1, batch_size, self.embed_size)
        
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False)
        out, h = self.gru(packed, h)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(out)
        out = self.dropout(out)
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
        
       
def train_doctorAI(model, trainset, validset, testset, args, optimizer, criterion):
    
    bestValidCrossEntropy = 1e20
    bestValidEpoch = 0
    testCrossEntropy = 0.0
    n_batches = int(np.ceil(float(len(trainset[0]))/float(args.batch_size)))
    
    print("Optimization Start!!")
    
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0

       
        for index in random.sample(range(n_batches), n_batches):
            
            batchX = trainset[0][index*args.batch:(index+1)*args.batch]
            batchY = trainset[1][index*args.batch:(index+1)*args.batch]       
            
            size = len(batchX)
            batchX, batchY, lengths = padMatrix(batchX, batchY, args.input_size, args.num_classes)
            
            optimizer.zero_grad()
            h = model.init_hidden(size)
            out, h = model(batchX.float(), lengths, h)
         
            batchY = batchY.to(dtype= torch.float)
            cost = criterion(out, batchY)
            cost.backward()
            optimizer.step()
            running_loss += cost.item()
         
        else:
            
            with torch.no_grad():
                model.eval()
                validAuc = calculate_auc(model, validset, args.batch, args.input_size, args.num_classes, criterion)
                print("Validation cross entropy:%f at epoch:%d" % (validAuc, epoch))
                if validAuc < bestValidCrossEntropy:
                    bestValidCrossEntropy = validAuc
                    bestValidEpoch = epoch
                    
                    testCrossEntropy = calculate_auc(model, testset, args.batch, args.input_size, args.num_classes, criterion)
                    
                    print('Test Cross Entropy:%f at Epoch:%d' % (testCrossEntropy, epoch))
                    torch.save(model, 'model.'+str(epoch)+'.pth')
                   
        print("Epoch:%d, Mean_Cost:%f" % (epoch, running_loss/n_batches))
            
    print('The Best Valid Cross Entropy:%f at epoch:%d' % (bestValidCrossEntropy, bestValidEpoch))
    print('The Test Cross Entropy: %f' % testCrossEntropy)
    
    
    

def parse_arguments(parser):
    parser.add_argument('--seq_file', type=str, required=True, help='path to the pickled file containing patient visit information') #seqeunce file
    parser.add_argument('--input_size', type=int, required=True, help='number of unique input medical codes') #n_input_codes
    parser.add_argument('--label_file', type=str, required=True, help='path to the pickled file containg patient label information') #label file
    parser.add_argument('--num_classes', type=int, required=True, help='number of unique classes')
    parser.add_argument('--out_file', type=str, required=True, help='path of directory the output models will be saved') #output file
    parser.add_argument('--emb_size', type=int, default=200, help='dimension size of embedding layer')
    parser.add_argument('--n_layers', type=int, default=2, help='total number of hidden layer')
    parser.add_argument('--hidden_dim', type=int, default=2000, help='hidden layer dimension size') #hidden dimension size
    parser.add_argument('--batch_size', type=int, default=100, help='batch size') #batch size
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs') #epochs
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate') #dropout
    
    args = parser.parse_args
    return args


def main(args):
    print("Loading Data ... ")
    trainset, validset, testset = load_data(args.seq_file, args.label_file)
    print("Done")
    
    print("Building Model ... ")
    model = GRUNet(args)
    model.init_weight()
    criterion = nn.BCELoss()
    optimizer = optim.Adadelta(model.parameters(), rho=0.95)
    
    
    print("Training Model ... ")
    train_doctorAI(model, trainset, validset, testset, args, optimizer, criterion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    main(args)