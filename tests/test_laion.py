import pandas as pd

from collections import Counter

from tqdm import tqdm

import string
import csv

import multiprocessing

import os
from datasets import load_dataset

import re

def count_words_rp():
    
    cnt = Counter()

    ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")

    for text in tqdm(ds["train"]["text"]):

        if (text):

            text = text.lower().strip()

            translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
            text = text.translate(translator)

            #print("%d: %s" % (index, text))

            for word in re.split(' |\n', text):
                word = word.strip()
                if (word):
                    cnt[word] += 1

        #if (index > 100):
        #    break

    most_common = cnt.most_common(10000)
    total = sum(cnt.values())

    return most_common, total

def count_words(p):

    cnt = Counter()

    file = 'E:\Research\Images\LAION400M\meta\part %05d 5b54c5d5 bbcf 484d a2ce 0d6f73df1a36 c000.snappy.parquet' % p
    print("opening %s" % file)

    df = pd.read_parquet(file, engine='pyarrow', columns=['URL', 'TEXT'])
    pd.read
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row["TEXT"]

        if (text):
            text = text.lower().strip()

            text = text.translate(str.maketrans('', '', string.punctuation))

            for word in text.split(" "):
                word = word.strip()
                if (word):
                    cnt[word] += 1

        #if (index > 1000):
        #    break

    most_common = cnt.most_common(10000)
    most_common_cnt = Counter()

    for row in most_common:
        most_common_cnt[row[0]] += row[1]

    total = sum(cnt.values())

    return most_common_cnt, total

#for p in range(0,32):
#    file = 'E:\Research\Images\LAION400M\meta\part %05d 5b54c5d5 bbcf 484d a2ce 0d6f73df1a36 c000.snappy.parquet' % p
#    print("opening %s" % file)

#    df = pd.read_parquet(file, engine='pyarrow', columns=['URL', 'TEXT'])

#    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#        text = row["TEXT"]

#        if (text):
#            text = text.lower().strip()

#            text = text.translate(str.maketrans('', '', string.punctuation))

#            for word in text.split(" "):
#                word = word.strip()
#                if (word):
#                    cnt[word] += 1

#most_common = cnt.most_common(50000)

if __name__ == '__main__':

    #pool_obj = multiprocessing.Pool(processes=8)
    #answer = pool_obj.map(count_words,range(0,32))

    #counter_merged, total_all = answer[0]

    #for counter, total in answer[1:]:
    #    for word in counter:
    #        counter_merged[word] += counter[word]

    #    total_all += total

    #most_common = counter_merged.most_common(10000)

    #total = sum(counter_merged.values())

    #with open('E:\Research\Images\LAION400M\most_common_10000_total_words_%d.csv' % total_all, 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile, delimiter=',')

    #    for row in most_common:
    #        try:
    #            writer.writerow([row[0], 100*row[1]/total_all])
    #        except:
                #pass

    most_common, total_all = count_words_rp()

    with open('E:\Research\Images\LAION400M\most_common_10000_total_words_%d_rp.csv' % total_all, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for row in most_common:
            try:
                writer.writerow([row[0], 100*row[1]/total_all])
            except:
                pass