# -*- coding: utf-8 -*-
import jieba
from util import DataManager
import pandas as pd
import os
import sys, argparse, os

parser = argparse.ArgumentParser(description='Sentiment classification')
#parser.add_argument('model')
parser.add_argument('--action', default='train')

#parser.add_argument('--train_path', default='data/training_label.txt', type=str)
#parser.add_argument('--test_path', default='data/testing_data.txt', type=str)
#parser.add_argument('--semi_path', default='data/training_nolabel.txt', type=str)

parser.add_argument('--d_base_dir', default='feature')

# training argument
parser.add_argument('--batch_size', default=32, type=float)
parser.add_argument('--nb_epoch', default=40, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.2, type=float)
parser.add_argument('--vocab_size', default=20000, type=int)
parser.add_argument('--max_length', default=50,type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=128, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.1,type=float)

# for testing
parser.add_argument('--test_y', dest='test_y', type=str, default='npy/1.npy')

# output path for your prediction
parser.add_argument('--result_path', default='result.csv')

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model/')
args = parser.parse_args()



def main():
    dm = DataManager()
    print ('Loading data...')
    if args.action == 'train':
        dm.add_data('train_data', args.d_base_dir, True)
    else:
        print ('Implement your testing parser')
        dm.add_data('test_data', args.d_base_dir, False)

    train_id = (dm.get_data('train_data')['train.question.id'])
    train_q = (dm.get_data('train_data')['train.question'])
    train_ans = (dm.get_data('train_data')['train.answer'])
    train_con = (dm.get_data('train_data')['train.context'])
    train_span = (dm.get_data('train_data')['train.span'])

    '''
    print (train_con)
    l = [jieba.cut(sentence, cut_all=False) for sentence in train_con]
    for i in l[3]:
        print (i,)
    '''
    word2vec_model = 'save/med250.model.bin'
    dm.sequence2matrix(word2vec_model)





if __name__ == '__main__':
    main()
