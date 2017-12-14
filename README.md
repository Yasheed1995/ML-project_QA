# Machine Learning Final Project

Delta Chinese QA
邁向中文問答之路

## Getting Started

Input: A short paragraph and a question
Output: A segment of paragraph


### NOTES

Git add ignore large files(save your own training data)
```
find . -size +90M | sed 's|^\./||g' | cat >> .gitignore; awk '!NF || !seen[$0]++' .gitignore
```

Jieba note


https://github.com/ldkrsi/jieba-zh_TW


Pre-trained wordvec


https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md


Kaggle


https://www.kaggle.com/c/ml-2017fall-final-chinese-qa/data


ppt

https://docs.google.com/presentation/d/1WQ2m6CbnCTkgUoDca782GPk9sqnCLxkc-hPxfg8y9p4/edit#slide=id.g29b893c7a1_0_57


### Evaluation 

The evaluation metric for this competition is Mean F1-Score. The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision p and recall r. Precision is the ratio of true positives (tp) to all predicted positives (tp + fp). Recall is the ratio of true positives to all actual positives (tp + fn). The F1 score is given by:

F1=2p⋅r/(p+r)  where  p=tp/(tp+fp),  r=tp/(tp+fn)

The F1 metric weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.

## Running the tests

Explain how to run the automated tests for this system

## Data Files

version : String
Data : Array
title : String
paragraphs : Array
context : String
qas : Array
question : String
id : uuid
answers : Arrays
answer_start : int
text : string


### references

seq2seq

http://blog.csdn.net/jerr__y/article/details/53749693


(ML 2017) 

http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/RNN%20(v2).pdf
http://cyruschiu.github.io/2017/02/24/learning-Tensoflow-Seq2Seq-for-translate/


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc