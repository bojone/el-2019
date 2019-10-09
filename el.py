#! -*- coding: utf-8 -*-

import json
from tqdm import tqdm
import os
import numpy as np
from random import choice
from itertools import groupby
from gensim.models import Word2Vec
import pyhanlp
from nlp_zero import Trie, DAG # pip install nlp_zero
import re


mode = 0
min_count = 2
char_size = 128
num_features = 3


word2vec = Word2Vec.load('../../kg/word2vec_baike/word2vec_baike')
id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}
word2id = {j: i for i, j in id2word.items()}
word2vec = word2vec.wv.syn0
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])


def tokenize(s):
    """如果pyhanlp不好用，自己修改tokenize函数，
    换成自己的分词工具即可。
    """
    return [i.word for i in pyhanlp.HanLP.segment(s)]


def sent2vec(S):
    """S格式：[[w1, w2]]
    """
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[(-1)].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


id2kb = {}
with open('../ccks2019_el/kb_data') as (f):
    for l in tqdm(f):
        _ = json.loads(l)
        subject_id = _['subject_id']
        subject_alias = list(set([_['subject']] + _.get('alias', [])))
        subject_alias = [alias.lower() for alias in subject_alias]
        object_regex = set(
            [i['object'] for i in _['data'] if len(i['object']) <= 10]
        )
        object_regex = sorted(object_regex, key=lambda s: -len(s))
        object_regex = [re.escape(i) for i in object_regex]
        object_regex = re.compile('|'.join(object_regex)) # 预先建立正则表达式，用来识别object是否在query出现过
        _['data'].append({
            'predicate': u'名称',
            'object': u'、'.join(subject_alias)
        })
        subject_desc = '\n'.join(
            u'%s：%s' % (i['predicate'], i['object']) for i in _['data']
        )
        subject_desc = subject_desc.lower()
        id2kb[subject_id] = {
            'subject_alias': subject_alias,
            'subject_desc': subject_desc,
            'object_regex': object_regex
        }


kb2id = {}
trie = Trie() # 根据知识库所有实体来构建Trie树

for i, j in id2kb.items():
    for k in j['subject_alias']:
        if k not in kb2id:
            kb2id[k] = []
            trie[k.strip(u'《》')] = 1
        kb2id[k].append(i)


def search_subjects(text_in):
    """实现最大匹配算法
    """
    R = trie.search(text_in)
    dag = DAG(len(text_in))
    for i, j in R:
        dag[(i, j)] = -1
    S = {}
    for i, j in dag.optimal_path():
        if text_in[i:j] in kb2id:
            S[(i, j)] = text_in[i:j]
    return S


train_data = []

with open('../ccks2019_el/train.json') as (f):
    for l in tqdm(f):
        _ = json.loads(l)
        train_data.append({
            'text': _['text'],
            'mention_data': [
                (x['mention'], int(x['offset']), x['kb_id'])
                for x in _['mention_data'] if x['kb_id'] != 'NIL'
            ]
        })


if not os.path.exists('../all_chars_me.json'):
    chars = {}
    for d in tqdm(iter(id2kb.values())):
        for c in d['subject_desc']:
            chars[c] = chars.get(c, 0) + 1
    for d in tqdm(iter(train_data)):
        for c in d['text'].lower():
            chars[c] = chars.get(c, 0) + 1
    chars = {i: j for i, j in chars.items() if j >= min_count}
    id2char = {i + 2: j for i, j in enumerate(chars)}
    char2id = {j: i for i, j in id2char.items()}
    json.dump([id2char, char2id], open('../all_chars_me.json', 'w'))
else:
    id2char, char2id = json.load(open('../all_chars_me.json'))


# 通过统计来精简词典，提高最大匹配的准确率
words_to_pred = {}
words_to_remove = {}
A, B, C = 1e-10, 1e-10, 1e-10
for d in train_data:
    R = set([(v, k[0]) for k, v in search_subjects(d['text']).items()])
    T = set([tuple(md[:2]) for md in d['mention_data']])
    A += len(R & T)
    B += len(R)
    C += len(T)
    R = set([i[0] for i in R])
    T = set([i[0] for i in T])
    for w in T:
        words_to_pred[w] = words_to_pred.get(w, 0) + 1
    for w in R - T:
        words_to_remove[w] = words_to_remove.get(w, 0) + 1


words = set(list(words_to_pred) + list(words_to_remove))
words = {
    i: (words_to_remove.get(i, 0) + 1.0) / (words_to_pred.get(i, 0) + 1.0)
    for i in words
}
words = {i: j for i, j in words.items() if j >= 5}

for w in words:
    del kb2id[w]
    trie[w] = 0


if not os.path.exists('../random_order_train.json'):
    random_order = range(len(train_data))
    np.random.shuffle(random_order)
    json.dump(random_order, open('../random_order_train.json', 'w'), indent=4)
else:
    random_order = json.load(open('../random_order_train.json'))


dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]


subjects = {}

for d in train_data:
    for md in d['mention_data']:
        if md[0] not in subjects:
            subjects[md[0]] = {}
        subjects[md[0]][md[2]] = subjects[md[0]].get(md[2], 0) + 1


candidate_links = {}

for k, v in subjects.items():
    for i, j in v.items():
        if j < 2:
            del v[i]
    if v:
        _ = set(v.keys()) & set(kb2id.get(k, []))
        if _:
            candidate_links[k] = list(_)


test_data = []

with open('../ccks2019_el/develop.json') as f:
    for l in tqdm(f):
        _ = json.loads(l)
        test_data.append(_)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
        for x in X
    ])


def isin_feature(text_a, text_b):
    y = np.zeros(len(''.join(text_a)))
    text_b = set(text_b)
    i = 0
    for w in text_a:
        if w in text_b:
            for c in w:
                y[i] = 1
                i += 1
    return y


def is_match_objects(text, object_regex):
    y = np.zeros(len(text))
    for i in object_regex.finditer(text):
        y[i.start():i.end()] = 1
    return y


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            np.random.shuffle(idxs)
            X1, X2, X1V, X2V, S1, S2, PRES1, PRES2, Y, T = (
                [], [], [], [], [], [], [], [], [], []
            )
            for i in idxs:
                d = self.data[i]
                text = d['text'].lower()
                text_words = tokenize(text)
                text = ''.join(text_words)
                x1 = [char2id.get(c, 1) for c in text]
                s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                mds = {}
                for md in d['mention_data']:
                    md = (md[0].lower(), md[1], md[2])
                    if md[0] in kb2id:
                        j1 = md[1]
                        j2 = j1 + len(md[0])
                        s1[j1] = 1
                        s2[j2 - 1] = 1
                        mds[(j1, j2)] = (md[0], md[2])
                if mds:
                    j1, j2 = choice(mds.keys())
                    y1 = np.zeros(len(text))
                    y1[j1:j2] = 1
                    x2 = choice(kb2id[mds[(j1, j2)][0]])
                    if x2 == mds[(j1, j2)][1]:
                        t = [1]
                    else:
                        t = [0]
                    object_regex = id2kb[x2]['object_regex']
                    x2 = id2kb[x2]['subject_desc']
                    x2_words = tokenize(x2)
                    x2 = ''.join(x2_words)
                    y2 = isin_feature(text, x2)
                    y3 = isin_feature(text_words, x2_words)
                    y4 = is_match_objects(text, object_regex)
                    y = np.vstack([y1, y2, y3, y4]).T
                    x2 = [char2id.get(c, 1) for c in x2]
                    pre_subjects = search_subjects(d['text'])
                    pres1, pres2 = np.zeros(len(text)), np.zeros(len(text))
                    for j1, j2 in pre_subjects:
                        pres1[j1] = 1
                        pres2[j2 - 1] = 1
                    X1.append(x1)
                    X2.append(x2)
                    X1V.append(text_words)
                    X2V.append(x2_words)
                    S1.append(s1)
                    S2.append(s2)
                    PRES1.append(pres1)
                    PRES2.append(pres2)
                    Y.append(y)
                    T.append(t)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        X1V = sent2vec(X1V)
                        X2V = sent2vec(X2V)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        PRES1 = seq_padding(PRES1)
                        PRES2 = seq_padding(PRES2)
                        Y = seq_padding(Y, np.zeros(1 + num_features))
                        T = seq_padding(T)
                        yield [X1, X2, X1V, X2V, S1, S2, PRES1, PRES2, Y, T], None
                        X1, X2, X1V, X2V, S1, S2, PRES1, PRES2, Y, T = (
                            [], [], [], [], [], [], [], [], [], []
                        )


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


class Attention(Layer):
    """多头注意力机制
    """
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(
            name='q_kernel',
            shape=(q_in_dim, self.out_dim),
            initializer='glorot_normal')
        self.k_kernel = self.add_weight(
            name='k_kernel',
            shape=(k_in_dim, self.out_dim),
            initializer='glorot_normal')
        self.v_kernel = self.add_weight(
            name='w_kernel',
            shape=(v_in_dim, self.out_dim),
            initializer='glorot_normal')
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = (None, None)
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


class MyBidirectional:
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer):
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
    def reverse_sequence(self, inputs):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        x, mask = inputs
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)
    def __call__(self, inputs):
        x, mask = inputs
        x_forward = self.forward_layer(x)
        x_backward = Lambda(self.reverse_sequence)([x, mask])
        x_backward = self.backward_layer(x_backward)
        x_backward = Lambda(self.reverse_sequence)([x_backward, mask])
        x = Concatenate()([x_forward, x_backward])
        x = Lambda(lambda x: x[0] * x[1])([x, mask])
        return x


x1_in = Input(shape=(None, ))
x2_in = Input(shape=(None, ))
x1v_in = Input(shape=(None, word_size))
x2v_in = Input(shape=(None, word_size))
s1_in = Input(shape=(None, ))
s2_in = Input(shape=(None, ))
pres1_in = Input(shape=(None, ))
pres2_in = Input(shape=(None, ))
y_in = Input(shape=(None, 1 + num_features))
t_in = Input(shape=(1, ))

x1, x2, x1v, x2v, s1, s2, pres1, pres2, y, t = (
    x1_in, x2_in, x1v_in, x2v_in, s1_in, s2_in, pres1_in, pres2_in, y_in, t_in
)

x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
x2_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x2)

embedding = Embedding(len(id2char) + 2, char_size)
dense = Dense(char_size, use_bias=False)

x1 = embedding(x1)
x1v = dense(x1v)
x1 = Add()([x1, x1v])
x1 = Dropout(0.2)(x1)

pres1 = Lambda(lambda x: K.expand_dims(x, 2))(pres1)
pres2 = Lambda(lambda x: K.expand_dims(x, 2))(pres2)
x1 = Concatenate()([x1, pres1, pres2])
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])

x1 = MyBidirectional(CuDNNLSTM(char_size // 2, return_sequences=True))([x1, x1_mask])

h = Conv1D(char_size, 3, activation='relu', padding='same')(x1)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)
ps1 = Lambda(lambda x: x[0] * x[1])([ps1, pres1]) # 这样一乘，相当于只从最大匹配的结果中筛选实体
ps2 = Lambda(lambda x: x[0] * x[1])([ps2, pres2]) # 这样一乘，相当于只从最大匹配的结果中筛选实体

s_model = Model([x1_in, x1v_in, pres1_in, pres2_in], [ps1, ps2])


x1 = Concatenate()([x1, y])
x1 = MyBidirectional(CuDNNLSTM(char_size // 2, return_sequences=True))([x1, x1_mask])
ys = Lambda(lambda x: K.sum(x[0] * x[1][..., :1], 1) / K.sum(x[1][..., :1], 1))([x1, y])

x2 = embedding(x2)
x2v = dense(x2v)
x2 = Add()([x2, x2v])
x2 = Dropout(0.2)(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])
x2 = MyBidirectional(CuDNNLSTM(char_size // 2, return_sequences=True))([x2, x2_mask])

x12 = Attention(8, 16)([x1, x2, x2, x2_mask, x1_mask])
x12 = Lambda(seq_maxpool)([x12, x1_mask])
x21 = Attention(8, 16)([x2, x1, x1, x1_mask, x2_mask])
x21 = Lambda(seq_maxpool)([x21, x2_mask])
x = Concatenate()([x12, x21, ys])
x = Dropout(0.2)(x)
x = Dense(char_size, activation='relu')(x)
pt = Dense(1, activation='sigmoid')(x)

t_model = Model([x1_in, x2_in, x1v_in, x2v_in, pres1_in, pres2_in, y_in], pt)


train_model = Model(
    [x1_in, x2_in, x1v_in, x2v_in, s1_in, s2_in, pres1_in, pres2_in, y_in, t_in],
    [ps1, ps2, pt]
)

s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)
s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * x1_mask) / K.sum(x1_mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * x1_mask) / K.sum(x1_mask)
pt_loss = K.mean(K.binary_crossentropy(t, pt))
loss = s1_loss + s2_loss + pt_loss

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()


class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]
    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))


EMAer = ExponentialMovingAverage(train_model)
EMAer.inject()


def extract_items(text_in):
    text_words = tokenize(text_in)
    text_old = ''.join(text_words)
    text_in = text_old.lower()
    _x1 = [char2id.get(c, 1) for c in text_in]
    _x1 = np.array([_x1])
    _x1v = sent2vec([text_words])
    pre_subjects = search_subjects(text_in)
    _pres1, _pres2 = np.zeros([1, len(text_in)]), np.zeros([1, len(text_in)])
    for j1, j2 in pre_subjects:
        _pres1[(0, j1)] = 1
        _pres2[(0, j2 - 1)] = 1
    _k1, _k2 = s_model.predict([_x1, _x1v, _pres1, _pres2])
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.4)[0], np.where(_k2 > 0.4)[0]
    _subjects = []
    for i in _k1:
        j = _k2[(_k2 >= i)]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i:j + 1]
            _subjects.append((_subject, i, j + 1))
    if _subjects:
        R = []
        _X2, _X2V, _Y = [], [], []
        _S, _IDXS = [], {}
        for _s in _subjects:
            _y1 = np.zeros(len(text_in))
            _y1[_s[1]: _s[2]] = 1
            if _s[0] in candidate_links:
                _IDXS[_s] = candidate_links.get(_s[0], [])
            else:
                _IDXS[_s] = kb2id.get(_s[0], [])
            for i in _IDXS[_s]:
                object_regex = id2kb[i]['object_regex']
                _x2 = id2kb[i]['subject_desc']
                _x2_words = tokenize(_x2)
                _x2 = ''.join(_x2_words)
                _y2 = isin_feature(text_in, _x2)
                _y3 = isin_feature(text_words, _x2_words)
                _y4 = is_match_objects(text_in, object_regex)
                _y = np.vstack([_y1, _y2, _y3, _y4]).T
                _x2 = [char2id.get(c, 1) for c in _x2]
                _X2.append(_x2)
                _X2V.append(_x2_words)
                _Y.append(_y)
                _S.append(_s)
        if _X2:
            _X2 = seq_padding(_X2)
            _X2V = sent2vec(_X2V)
            _Y = seq_padding(_Y, np.zeros(1 + num_features))
            _X1 = np.repeat(_x1, len(_X2), 0)
            _X1V = np.repeat(_x1v, len(_X2), 0)
            _PRES1 = np.repeat(_pres1, len(_X2), 0)
            _PRES2 = np.repeat(_pres2, len(_X2), 0)
            scores = t_model.predict([_X1, _X2, _X1V, _X2V, _PRES1, _PRES2, _Y])[:, 0]
            for k, v in groupby(zip(_S, scores), key=lambda s: s[0]):
                ks = _IDXS[k]
                vs = [j[1] for j in v]
                if np.max(vs) < 0.1:
                    continue
                kbid = ks[np.argmax(vs)]
                R.append((text_old[k[1]:k[2]], k[1], kbid))
        return R
    else:
        return []


def test(test_data):
    F = open('result.json', 'w')
    for d in tqdm(iter(test_data)):
        d['mention_data'] = [
            dict(zip(['mention', 'offset', 'kb_id'], [md[0], str(md[1]), md[2]]))
            for md in set(extract_items(d['text']))
        ]
        F.write(json.dumps(d, ensure_ascii=False).encode('utf-8') + '\n')
    F.close()


class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.0
    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('best_model.weights')
        print 'f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best)
        EMAer.reset_old_weights()
    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10
        F = open('dev_pred.json', 'w')
        pbar = tqdm()
        for d in dev_data:
            R = set(extract_items(d['text']))
            T = set(d['mention_data'])
            A += len(R & T)
            B += len(R)
            C += len(T)
            s = json.dumps(
                {
                    'text': d['text'],
                    'mention_data': list(T),
                    'mention_data_pred': list(R),
                    'new': list(R - T),
                    'lack': list(T - R)
                },
                ensure_ascii=False,
                indent=4)
            F.write(s.encode('utf-8') + '\n')
            pbar.update(1)
            f1, pr, rc = 2 * A / (B + C), A / B, A / C
            pbar.set_description('< f1: %.4f, precision: %.4f, recall: %.4f >' % (f1, pr, rc))
        F.close()
        pbar.close()
        return (2 * A / (B + C), A / B, A / C)


evaluator = Evaluate()
train_D = data_generator(train_data)

if __name__ == '__main__':
    train_model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=120,
        callbacks=[evaluator]
    )
else:
    train_model.load_weights('best_model.weights')
