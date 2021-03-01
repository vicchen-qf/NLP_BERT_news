#snow sentiment
from __future__ import unicode_literals
import normal
import seg
import sys
import gzip
import utils
import marshal
from sim import bm25
from math import log, exp
from utils.frequency import AddOneProb

#bert qa
import os
import re
import json
import math
import string
import numpy as np

#run tensorflow on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertConfig, TFBertModel

#句子最大長度
max_len = 384
configuration = BertConfig()

##BERT pre-trian模型、QA模型、QA主題等設定
model_version = 'bert-base-chinese'
# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained(model_version)
save_path = "bert_base_chinese/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

#直接吃bert-base-chinese裡面的vocab.txt
tokenizer = BertWordPieceTokenizer("bert_base_chinese/vocab.txt", lowercase=True)

##如果要更改QA模型，改以下路徑即可
qa_model_path = './model/model_QA_all/all_topic_weights.h5'

#目前模型包含17個主題，test ACC:0.87
#目前已訓練完成的主題，用來看標題是否出現這些主題，並問問題
#未來如果直接用key word去搜，所有的keyword更新在這邊即可。
entity = ['營運', '訂單', '營收', '需求', '業績', 'EPS', '出貨', '股價', '獲利', '外資', '銷售', '生產', '產能', '動能', 
          '淨利', '毛利', '純益']

##要進入BERT QA前，處理標題、問題的embeddings部分
#encoding
class SquadExample:
    def __init__(self, question, context):
        self.question = question
        self.context = context
        self.skip = False
        
    #資料預處理
    def preprocess(self):
        context = self.context
        question = self.question

        # Tokenize context
        tokenized_context = tokenizer.encode(context)
       
        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.context_token_to_char = tokenized_context.offsets

##一個title可能有多個QA，先用" # "來切割
##再丟進去SquadExample進行資料前處理、encoding
def create_squad_examples(t, q):
    squad_examples = []
    context = t
    questions = re.split('#', q)
    for i in range(len(questions)):
        question = questions[i]
        squad_eg = SquadExample(question, context)
        squad_eg.preprocess()
        squad_examples.append(squad_eg)
    return squad_examples

##建構(input id、token type id、attention mask)X 
def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    return x

##建構TFBert模型
def create_model():
    # BERT encoder
    encoder = TFBertModel.from_pretrained('bert-base-chinese')
    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)
    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=5e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    model.load_weights(qa_model_path)

    return model

##資料清理(空格改成句號，句尾加句號)
def clean_titles(text):
    text = text.replace(" ", "。")
    text = text.replace("　","。")
    text += '。'
    return text

##資料清理(去除答案包含數字的內容)
def clean_qa_word(text):
    text = re.sub(u'[▽△▼▲]+.\w+：', '', text)
    text = re.sub(u'[【《(（〈]+.*?[】》)）〉]+', '', text)
    text = re.sub(r'\d+\.?\d*[兆\億\萬\千\百][元]', '', text)
    text = re.sub(r'\d+\.?\d*[兆\億\萬\千\百]', '', text)
    text = re.sub(r'[-+]\d+\.?\d*[元\倍\成]', '', text)
    text = re.sub(r'\d+\.?\d*[元\倍\成]', '', text)
    text = re.sub(r'[-+]\d+\.?\d*[%％]', '', text)
    text = re.sub(r'\d+\.?\d*[%％]', '', text)
    text = str.strip(re.sub(u'\s*?\w+：', ' ', text))
    return text

#清理重複、空白的答案 ex:(上季需求持續喊燒，穩懋1月營收年增##持續喊燒#年增 -> 只留:持續喊燒#年增)
def qa_drop_dup(x):
    x = re.split('#', x)
    #drop empty string
    x = list(filter(None, x))
    #drop 重複的 string
    x = list(dict.fromkeys(x))
    #drop 比較長的重複的string
    drop_word = []
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                if (x[i] in x[j]) and (x[j] not in drop_word):
                    drop_word.append(x[j])
    for i in drop_word:
        x.remove(i)
    word = ''
    for i in x:
        word += (i + '#')
    if len(word) == 0:
        word = '#'
    else:
        word = word[:len(word) - 1]
    return word

##create model using CPU
model = create_model()

##預測QA之答案
##包含建構問題、建構embedding、預測過程
def create_qa_word(news, model, entity):
    #data cleaning
    title = clean_titles(news['titles'])
    #prepare QA
    question = ''
    for i in entity:
        if i in title:
            if len(question) == 0:
                question = i + '表現如何?'
            else:
                question += ('#' + i + '表現如何?')
    QA_word = ''
    #如果標題不存在entity -> 維持整句標題
    if len(question) == 0:
        QA_word = clean_qa_word(title)
    #標籤為中立的新聞，不經過QA
    elif news['label'] == '1':
        QA_word = ''
    #標題有entity -> 尋找狀態詞彙
    else:
        all_squad_examples = create_squad_examples(title, question)
        x_all = create_inputs_targets(all_squad_examples)
    
        #predict
        pred_start, pred_end = model.predict(x_all)
        word_list = ''
        all_examples_no_skip = [_ for _ in all_squad_examples if _.skip == False]

        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = all_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
                word_list += (pred_ans + '#')
            else:
                pred_ans = squad_eg.context[pred_char_start:]
                word_list += (pred_ans + '#')
        #之所以要取到前一個是因為最後一個字元會是"#"，將其去除
        QA_word = word_list[:len(word_list)-1]
        if len(QA_word) == 0:
            QA_word = squad_eg.context
            
        #清掉數字
        QA_word = clean_qa_word(QA_word)
        if len(QA_word) == 0:
            QA_word = '#'
    #清掉重複的句子、空白
    QA_word = qa_drop_dup(QA_word)
    news.update({'QA_word': QA_word})
    return news

##naive bayes(單純貝氏)
class Bayes(object):

    def __init__(self):
        self.d = {}
        self.total = 0
    #讀取pre-train好的貝氏模型
    def load(self, fname, iszip=True):
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            d = marshal.load(open(fname, 'rb'))
        else:
            try:
                f = gzip.open(fname, 'rb')
                d = marshal.loads(f.read())
            except IOError:
                f = open(fname, 'rb')
                d = marshal.loads(f.read())
            f.close()
        self.total = d['total']
        self.d = {}
        for k, v in d['d'].items():
            self.d[k] = AddOneProb()
            self.d[k].__dict__ = v

    def classify(self, x):
        tmp = {}
        for k in self.d:
            tmp[k] = log(self.d[k].getsum()) - log(self.total)
            for word in x:
                tmp[k] += log(self.d[k].freq(word))
        ret, prob = 0, 0
        for k in self.d:
            now = 0
            try:
                for otherk in self.d:
                    now += exp(tmp[otherk]-tmp[k])
                now = 1/now
            except OverflowError:
                now = 0
            if now > prob:
                ret, prob = k, now
        return (ret, prob)

##snow neg(負面情感分析器)
class Sentiment_neg(object):

    def __init__(self):
        self.classifier = Bayes()

    def load(self, fname='./model/all_neg.marshal', iszip=True):
        self.classifier.load(fname, iszip)

    def handle(self, doc):
        words = seg.seg(doc)
        words = normal.filter_stop(words)
        return words

    def classify(self, sent):
        ret, prob = self.classifier.classify(self.handle(sent))
        if ret == 'pos':
            return prob
        return 1-prob


classifier_neg = Sentiment_neg()
classifier_neg.load()

def classify_neg(sent):
    return classifier_neg.classify(sent)

##snow pos(正面情感分析器)
class Sentiment_pos(object):

    def __init__(self):
        self.classifier = Bayes()

    def load(self, fname='./model/all_pos.marshal', iszip=True):
        self.classifier.load(fname, iszip)

    def handle(self, doc):
        words = seg.seg(doc)
        words = normal.filter_stop(words)
        return words

    def classify(self, sent):
        ret, prob = self.classifier.classify(self.handle(sent))
        if ret == 'pos':
            return prob
        return 1-prob


classifier_pos = Sentiment_pos()
classifier_pos.load()

def classify_pos(sent):
    return classifier_pos.classify(sent)

class SnowNLP(object):

    def __init__(self, doc):
        self.doc = doc
        self.bm25 = bm25.BM25(doc)

    @property
    def words(self):
        return seg.seg(self.doc)

    @property
    def sentences(self):
        return normal.get_sentences(self.doc)

    @property
    def sentiments_pos(self):
        return classify_pos(self.doc)
    
    @property
    def sentiments_neg(self):
        return classify_neg(self.doc)

##建構QA、將狀態詞彙打分數
def sent_score(news_dict):
    #create QA word
    news = create_qa_word(news_dict, model, entity)
    #sentiment
    #neg
    if news['label'] == '0':
        s = SnowNLP(news['QA_word'])
        score = -0.5 - s.sentiments_neg/2
        news.update({'score': score})
    #neu = 0
    elif news['label'] == '1':
        news.update({'score': 0})
    #pos
    elif news['label'] == '2':
        s = SnowNLP(news['QA_word'])
        score = 0.5 + s.sentiments_pos/2
        news.update({'score': score})
    return news

##一次批讀整個data frame，並回傳打完分數的df
def update_score(data):
    data['QA_word'] = 0
    data['score'] = 0
    for index, row in data.iterrows():
        news = {'titles': row['titles'],
               'label': str(row['predict_label'])}
        news = sent_score(news)
        data['QA_word'].iloc[index] = news['QA_word']
        data['score'].iloc[index] = news['score']
    return data