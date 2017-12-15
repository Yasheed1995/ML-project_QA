# -*- coding: utf-8 -*-
import logging
import sys

from gensim.corpora import WikiCorpus

def main():

    #if len(sys.argv) != 2:
        #print("Usage: python3 " + sys.argv[0] + " wiki_data_path")
        #exit()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    wiki_corpus_ = []
    for i in range(1):
        wiki_corpus_.append(WikiCorpus(sys.argv[i+1], dictionary={}))
    
    texts_num = 0
    filename = 'save/wiki_texts'
    
    with open("save/wiki_texts_1.txt",'w',encoding='utf-8') as output:
        for cor in wiki_corpus_:
            for text in cor.get_texts():
                output.write(' '.join(text) + '\n')
                texts_num += 1
                if texts_num % 10000 == 0:
                    logging.info("已處理 %d 篇文章" % texts_num)

if __name__ == "__main__":
    main()
