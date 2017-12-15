# -*- coding: utf-8 -*-
import jieba
from util import DataManager
import pandas as pd
import os
import sys, argparse, os
from gensim.models import word2vec
from gensim import models
import logging

parser = argparse.ArgumentParser(description='Final Project QA')
#parser.add_argument('model')
parser.add_argument('--action', default='train')

#parser.add_argument('--train_path', default='data/training_label.txt', type=str)
#parser.add_argument('--test_path', default='data/testing_data.txt', type=str)

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
    dm.add_data('train_data', 'feature', True)
    dm.add_data('test_data', 'feature', False)
    
    print ('getting data...')
    train_id = (dm.get_data('train_data')['train.question.id'])
    train_q = (dm.get_data('train_data')['train.question'])
    train_ans = (dm.get_data('train_data')['train.answer'])
    train_con = (dm.get_data('train_data')['train.context'])
    train_span = (dm.get_data('train_data')['train.span'])
    
    test_id = (dm.get_data('test_data')['test.question.id'])
    test_q = (dm.get_data('test_data')['test.question'])
    test_con = (dm.get_data('test_data')['test.context'])

    print ('loading model...')
    word2vec_model = models.Word2Vec.load('save/med250.model.bin')

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    '''
    for i in range(1):
        print (train_con[i])
    print (len(train_con))
    
    for word in train_con[0]:
        try:
            print (word2vec_model[word])
        except KeyError:
            print ('EEEEEEEE')
    
    l = [jieba.cut(sentence, cut_all=False) for sentence in train_con]
    print ('|'.join(l[0]))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    vec_matrix = []
    
    try:
        for context in test_con:
           
            tmp = []
            words = list(jieba.cut(context, cut_all=False))
            for i in range(300):
                if i < len(words):
                    try:
                        tmp.append(word2vec_model[words[i]])
                    except KeyError:
                        #print ("%s not found" % word)
                        tmp.append(np.zeros(shape=(250,)))
                else:
                    tmp.append(np.zeros(shape=(250,)))
            tmp = np.array(tmp)
            
            vec_matrix.append(tmp)
        vec_matrix = np.array(vec_matrix)
        print (vec_matrix)
        for i in range(100):
            print (vec_matrix[i].shape)
        
    except KeyError:
        pass
    '''
    
    print ('seq 2 matrix...')
    dm.sequence2matrix(word2vec_model)

    print ('initial model...')
    model = build_model(args)
    print (model.summary())

    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning: load an exist model and keep training')
            model.load_weights('model/model.h5')
        else:
            pass
    else args.action == 'test':
        print ('Warning: testing without loading any model')

    if args.action == 'train':
        pass
        # define input, input shape

        # earlystop

        # save model path

        # checkpoint

        # fit 

        # save history

    elif args.action == 'test':
        pass


    

if __name__ == '__main__':
    main()
