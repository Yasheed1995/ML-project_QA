from gensim.models import word2vec
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("save/wiki_seg_1.txt")
    model = word2vec.Word2Vec(sentences, size=250, min_count=2)

    #保存模型，供日後使用
    model.save("save/med250.model.bin")

    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model.bin")

if __name__ == "__main__":
    main()
