 # -*- coding: utf-8 -*-
import jieba
import os

path = 'feature'

print os.listdir(path)
'''
for i in os.listdir(path):
	path = path + '/' + i
	print ('read data from %s...'%path)
	print i.split('.')[0]
	path = 'feature'
'''
sentences = ["獨立音樂需要大家一起來推廣，歡迎加入我們的行列！", "我沒有真實的自我"]
print "Input：", sentences
l = [jieba.cut(sentence, cut_all=False) for sentence in sentences]
#words = jieba.cut(sentence, cut_all=False)
print "Output 精確模式 Full Mode："
for i in l[0]:
	print i,
