# -*- coding: utf-8 -*-

import codecs
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

frequency = [3,5]

flag_global = dict()
result = []
file_names = ["../data/data_douban/train_mt.raw"]#,"../data/data_twitter/train_mt.raw"]#,"../data/data_amazon/train_mt.raw","../data/data_tieba/train_mt.raw","../data/data_twitter/train_mt.raw"]
for index,file_name in enumerate(file_names):
    count = dict()
    f = codecs.open(file_name, 'r', 'utf-8').readlines()
    for line in f:
        line = line.split("######")
        try:
            line = line[0].split()+line[1].split()
        except Exception, e:
            print Exception,e
            print line
            print file_name
            exit()
        for j in line:
            if j in count:
                count[j] = count[j]+1
            else:
                count[j] = 1

    for item in count.keys():
        #print item
        if count[item]>frequency[index]: 
            flag_global[item]=1

for i in flag_global.keys():
    result.append(i)
f_w = codecs.open("models/vec100.txt", 'w', 'utf-8')
f_w.write(str(len(result))+" "+"100\n")
str_emb = ""
for i in range(100):
    str_emb += "0"
embedding = " ".join(str_emb)
print len(result)
for i in result:
    #print i
    xx = str(i).encode('utf-8')+" ".encode('utf-8')+embedding.encode('utf-8')+"\n".encode('utf-8')
    xx = xx.encode('utf-8')
    f_w.write(xx)
print len(result)
