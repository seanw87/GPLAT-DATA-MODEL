import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
import tqdm

print(tf.__version__)


# 数据加载
# 正向情绪
with open("data/pos_feedbacks_seg.txt", "r", encoding="utf-8") as f:
    pos_text = f.read()
# 负面情绪
with open("data/neg_feedbacks_seg.txt", "r", encoding="utf-8") as f:
    neg_text = f.read()


# 描述性统计
# 正向文本统计
print("-" * 20 + " POSITIVE TEXT " + "-" * 20)
# 分句
pos_sentences = pos_text.lower().split("\n")
print("Total positive sentences: {}".format(len(pos_sentences)))
print("The average length of positive sentences: {}".format(np.mean([len(sentence.split("|")) for sentence in pos_sentences])))
print("The max length of positive sentences: {}".format(np.max([len(sentence.split("|")) for sentence in pos_sentences])))
print("The min length of positive sentences: {}".format(np.min([len(sentence.split("|")) for sentence in pos_sentences])))
# 统计高频词
c = Counter(pos_text.split("|")).most_common(100)
print("Most common words in positive sentences: \n{}".format(c))

# 负面文本统计
print()
print("-" * 20 + " NEGATIVE TEXT " + "-" * 20)
# 分句
neg_sentences = neg_text.lower().split("\n")
print("Total negative sentences: {}".format(len(neg_sentences)))
print("The average length of negative sentences: {}".format(np.mean([len(sentence.split("|")) for sentence in neg_sentences])))
print("The max length of negative sentences: {}".format(np.max([len(sentence.split("|")) for sentence in neg_sentences])))
print("The min length of negative sentences: {}".format(np.min([len(sentence.split("|")) for sentence in neg_sentences])))
# 统计高频词
c = Counter(neg_text.split("|")).most_common(100)
print("Most common words in negative sentences: \n{}".format(c))


# 数据预处理
# 句子最大长度
SENTENCE_LIMIT_SIZE = 20
# 合并pos和neg文本
total_text = pos_text + "\n" + neg_text
# 统计词汇
c = Counter(total_text.split("|"))
# 倒序查看词频
# 实际中一般在分词之后会对单词进行词干化（Stem）处理，之后再进行词频统计
sorted(c.most_common(), key=lambda x: x[1])
# 初始化两个token：pad和unk
vocab = ["<pad>", "<unk>"]
# 去除出现频次为1次的单词
for w, f in c.most_common():
    if f > 1:
        vocab.append(w)

print("The total size of our vocabulary is: {}".format(len(vocab)))

# 构造映射
# 单词到编码的映射，例如machine -> 10283
word_to_token = {word: token for token, word in enumerate(vocab)}
# 编码到单词的映射，例如10283 -> machine
token_to_word = {token: word for word, token in word_to_token.items()}


# 转换文本
def convert_text_to_token(sentence, word_to_token_map=word_to_token, limit_size=SENTENCE_LIMIT_SIZE):
    """
    根据单词-编码映射表将单个句子转化为token
    @param sentence: 句子，str类型
    @param word_to_token_map: 单词到编码的映射
    @param limit_size: 句子最大长度。超过该长度的句子进行截断，不足的句子进行pad补全
    return: 句子转换为token后的列表
    """
    # 获取unknown单词和pad的token
    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]
    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word, unk_id) for word in sentence.lower().split("|")]
    # Pad
    if len(tokens) < limit_size:
        #################################################
        ####################notice#######################
        #################################################
        # tokens.extend([0] * (limit_size - len(tokens)))
        tokens.extend([pad_id] * (limit_size - len(tokens)))
    # Trunc
    else:
        tokens = tokens[:limit_size]
    return tokens

# 对pos文本处理
pos_tokens = []
for sentence in tqdm.tqdm(pos_sentences):
    tokens = convert_text_to_token(sentence)
    pos_tokens.append(tokens)
# 对neg文本处理
neg_tokens = []
for sentence in tqdm.tqdm(neg_sentences):
    tokens = convert_text_to_token(sentence)
    neg_tokens.append(tokens)

# 转化为numpy格式，方便处理
pos_tokens = np.array(pos_tokens)
neg_tokens = np.array(neg_tokens)
# 合并所有语料
total_tokens = np.concatenate((pos_tokens, neg_tokens), axis=0)
print("The shape of all tokens in our corpus: ({}, {})".format(*total_tokens.shape))
# 转化为numpy格式，方便处理
pos_targets = np.ones((pos_tokens.shape[0]))
neg_targets = np.zeros((neg_tokens.shape[0]))
# 合并所有target
total_targets = np.concatenate((pos_targets, neg_targets), axis=0).reshape(-1, 1)

print("The shape of all targets in our corpus: ({}, {})".format(*total_targets.shape))


# 构造词向量
with open("data/wordvectors/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5", 'r', encoding="utf-8") as f:
    words = set()
    word_to_vec = {}
    for line in f:
        line = line.strip().split()
        # 当前单词
        curr_word = line[0]
        words.add(curr_word)
        # 当前词向量
        word_to_vec[curr_word] = np.array(line[1:], dtype=np.float32)
print("The number of words which have pretrained-vectors in vocab is: {}".format(len(set(vocab)&set(words))))
print()
print("The number of words which do not have pretrained-vectors in vocab is : {}".format(len(set(vocab))-
                                                                                         len(set(vocab)&set(words))))



# 构造词向量矩阵
VOCAB_SIZE = len(vocab)
EMBEDDING_SIZE = 300

# 初始化词向量矩阵（这里命名为static是因为这个词向量矩阵用预训练好的填充，无需重新训练）
static_embeddings = np.zeros([VOCAB_SIZE, EMBEDDING_SIZE])

for word, token in tqdm.tqdm(word_to_token.items()):
    # 用词向量填充，如果没有对应的词向量，则用随机数填充
    word_vector = word_to_vec.get(word, 0.2 * np.random.random(EMBEDDING_SIZE) - 0.1)
    static_embeddings[token, :] = word_vector

# 重置PAD为0向量
pad_id = word_to_token["<pad>"]
static_embeddings[pad_id, :] = np.zeros(EMBEDDING_SIZE)
static_embeddings = static_embeddings.astype(np.float32)




# 辅助函数
# 分割train和test
def split_train_test(x, y, train_ratio=0.8, shuffle=True):
    """
    分割train和test
    @param x: 输入特征
    @param y: 目标
    @param train_ratio: 训练样本比例
    @param shuffle: 是否shuffle
    """
    assert x.shape[0] == y.shape[0], print("error shape!")
    if shuffle:
        # shuffle
        shuffled_index = np.random.permutation(range(x.shape[0]))
        x = x[shuffled_index]
        y = y[shuffled_index]
    # 分离train和test
    train_size = int(x.shape[0] * train_ratio)
    x_train = x[:train_size]
    x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    return x_train, x_test, y_train, y_test

# 划分train和test
x_train, x_test, y_train, y_test = split_train_test(total_tokens, total_targets)

# get_batch函数
BATCH_SIZE = 256
def get_batch(x, y, batch_size=BATCH_SIZE, shuffle=True):
    assert x.shape[0] == y.shape[0], print("error shape!")
    # shuffle
    if shuffle:
        shuffled_index = np.random.permutation(range(x.shape[0]))
        x = x[shuffled_index]
        y = y[shuffled_index]

    # 统计共几个完整的batch
    n_batches = int(x.shape[0] / batch_size)
    for i in range(n_batches - 1):
        x_batch = x[i * batch_size: (i + 1) * batch_size]
        y_batch = y[i * batch_size: (i + 1) * batch_size]

        yield x_batch, y_batch




###########################################################################################
###########################################################################################
###########################################################################################
# DNN模型
tf.reset_default_graph()

# 定义神经网络超参数
HIDDEN_SIZE = 512
LEARNING_RATE = 0.001
EPOCHES = 50
BATCH_SIZE = 256

with tf.name_scope("dnn"):
    # 输入及输出tensor
    with tf.name_scope("placeholders"):
        inputs = tf.placeholder(dtype=tf.int32, shape=(None, SENTENCE_LIMIT_SIZE), name="inputs")
        targets = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")

    # embeddings
    with tf.name_scope("embeddings"):
        # 用pre-trained词向量来作为embedding层
        embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
        embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")
        # 相加词向量得到句子向量
        sum_embed = tf.reduce_sum(embed, axis=1, name="sum_embed")

    # model
    with tf.name_scope("model"):
        # 隐层权重
        W1 = tf.Variable(tf.random_normal(shape=(EMBEDDING_SIZE, HIDDEN_SIZE), stddev=0.1), name="W1")
        b1 = tf.Variable(tf.zeros(shape=(HIDDEN_SIZE), name="b1"))

        # 输出层权重
        W2 = tf.Variable(tf.random_normal(shape=(HIDDEN_SIZE, 1), stddev=0.1), name="W2")
        b2 = tf.Variable(tf.zeros(shape=(1), name="b2"))

        # 结果
        z1 = tf.add(tf.matmul(sum_embed, W1), b1)
        a1 = tf.nn.relu(z1)

        logits = tf.add(tf.matmul(a1, W2), b2)
        outputs = tf.nn.sigmoid(logits, name="outputs")

    # loss
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))

    # optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # evaluation
    with tf.name_scope("evaluation"):
        correct_preds = tf.equal(tf.cast(tf.greater(outputs, 0.5), tf.float32), targets)
        accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=1))

# 训练模型
# 存储准确率
dnn_train_accuracy = []
dnn_test_accuracy = []

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./graphs/dnn", tf.get_default_graph())

    n_batches = int(x_train.shape[0] / BATCH_SIZE)

    for epoch in range(EPOCHES):
        total_loss = 0

        for x_batch, y_batch in get_batch(x_train, y_train):
            _, batch_loss = sess.run([optimizer, loss],
                                     feed_dict={inputs: x_batch, targets: y_batch})

            total_loss += batch_loss

        # 在train上准确率
        train_corrects = sess.run(accuracy, feed_dict={inputs: x_train, targets: y_train})
        train_acc = train_corrects / x_train.shape[0]
        dnn_train_accuracy.append(train_acc)

        # 在test上准确率
        test_corrects = sess.run(accuracy, feed_dict={inputs: x_test, targets: y_test})
        test_acc = test_corrects / x_test.shape[0]
        dnn_test_accuracy.append(test_acc)

        print("Epoch: {}, Train loss: {:.4f}, Train accuracy: {:.4f}, Test accuracy: {:.4f}".format(epoch + 1,
                                                                                                    total_loss / n_batches,
                                                                                                    train_acc,
                                                                                                    test_acc))
    # 存储模型
    saver.save(sess, "./checkpoints/dnn")
    writer.close()

plt.plot(dnn_train_accuracy)
plt.plot(dnn_test_accuracy)
plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of DNN model")
plt.legend(["train", "test"])

# 预测模型
# 在test上的准确率
with tf.Session() as sess:
    saver.restore(sess, "checkpoints/dnn")

    total_correct = 0
    acc = sess.run(accuracy,
                   feed_dict={inputs: x_test,
                              targets: y_test})
    total_correct += acc
    print("The DNN model accuracy on test set: {:.2f}%".format(100 * total_correct / x_test.shape[0]))

# 在命令行执行tensorboard --logdir="./graphs/dnn" --port 6006可以看到模型的tensorboard




###########################################################################################
###########################################################################################
###########################################################################################
# RNN模型
tf.reset_default_graph()

# 定义网络超参数
HIDDEN_SIZE = 512
LEARNING_RATE = 0.001
KEEP_PROB = 0.5
EPOCHES = 50
BATCH_SIZE = 256

with tf.name_scope("rnn"):
    # placeholders
    with tf.name_scope("placeholders"):
        inputs = tf.placeholder(dtype=tf.int32, shape=(None, 20), name="inputs")
        targets = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")

    # embeddings
    with tf.name_scope("embeddings"):
        embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
        embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")

    # model
    with tf.name_scope("model"):
        # 构造lstm单元
        lstm = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        # 添加dropout
        drop_lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=KEEP_PROB)
        _, lstm_state = tf.nn.dynamic_rnn(drop_lstm, embed, dtype=tf.float32)

        # 输出层权重
        W = tf.Variable(tf.truncated_normal((HIDDEN_SIZE, 1), mean=0.0, stddev=0.1), name="W")
        b = tf.Variable(tf.zeros(1), name="b")

        logits = tf.add(tf.matmul(lstm_state.h, W), b)
        outputs = tf.nn.sigmoid(logits, name="outputs")

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))

    # optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # evaluation
    with tf.name_scope("evaluation"):
        correct_preds = tf.equal(tf.cast(tf.greater(outputs, 0.5), tf.float32), targets)
        accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=1))

# 训练模型
# 存储准确率
rnn_train_accuracy = []
rnn_test_accuracy = []

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./graphs/rnn", tf.get_default_graph())
    n_batches = int(x_train.shape[0] / BATCH_SIZE)

    for epoch in range(EPOCHES):
        total_loss = 0
        for x_batch, y_batch in get_batch(x_train, y_train):
            _, l = sess.run([optimizer, loss],
                            feed_dict={inputs: x_batch,
                                       targets: y_batch})
            total_loss += l

        train_corrects = sess.run(accuracy, feed_dict={inputs: x_train, targets: y_train})
        train_acc = train_corrects / x_train.shape[0]
        rnn_train_accuracy.append(train_acc)

        test_corrects = sess.run(accuracy, feed_dict={inputs: x_test, targets: y_test})
        test_acc = test_corrects / x_test.shape[0]
        rnn_test_accuracy.append(test_acc)

        print("Training epoch: {}, Train loss: {:.4f}, Train accuracy: {:.4f}, Test accuracy: {:.4f}".format(epoch + 1,
                                                                                                             total_loss / n_batches,
                                                                                                             train_acc,
                                                                                                             test_acc))

    saver.save(sess, "checkpoints/rnn")
    writer.close()

plt.plot(rnn_train_accuracy)
plt.plot(rnn_test_accuracy)
plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of LSTM model")
plt.legend(["train", "test"])

# 预测模型
# 在test上的准确率
with tf.Session() as sess:
    saver.restore(sess, "checkpoints/rnn")

    total_correct = sess.run(accuracy,
                             feed_dict={inputs: x_test, targets: y_test})

    print("The LSTM model accuracy on test set: {:.2f}%".format(100 * total_correct / x_test.shape[0]))

# 在命令行执行tensorboard --logdir="./graphs/rnn" --port 6006可以看到模型的tensorboard




###########################################################################################
###########################################################################################
###########################################################################################
# CNN模型

tf.reset_default_graph()

# 我在这里定义了5种filter，每种100个
filters_size = [2, 3, 4, 5, 6]
num_filters = 100
# 超参数
BATCH_SIZE = 256
EPOCHES = 50
LEARNING_RATE = 0.001
L2_LAMBDA = 10
KEEP_PROB = 0.8

# 构建模型图
with tf.name_scope("cnn"):
    with tf.name_scope("placeholders"):
        inputs = tf.placeholder(dtype=tf.int32, shape=(None, 20), name="inputs")
        targets = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")

    # embeddings
    with tf.name_scope("embeddings"):
        embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
        embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")
        # 添加channel维度
        embed_expanded = tf.expand_dims(embed, -1, name="embed_expand")

    # 用来存储max-pooling的结果
    pooled_outputs = []

    # 迭代多个filter
    for i, filter_size in enumerate(filters_size):
        with tf.name_scope("conv_maxpool_%s" % filter_size):
            filter_shape = [filter_size, EMBEDDING_SIZE, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, mean=0.0, stddev=0.1), name="W")
            b = tf.Variable(tf.zeros(num_filters), name="b")

            conv = tf.nn.conv2d(input=embed_expanded,
                                filter=W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")

            # 激活
            a = tf.nn.relu(tf.nn.bias_add(conv, b), name="activations")
            # 池化
            max_pooling = tf.nn.max_pool(value=a,
                                         ksize=[1, SENTENCE_LIMIT_SIZE - filter_size + 1, 1, 1],
                                         strides=[1, 1, 1, 1],
                                         padding="VALID",
                                         name="max_pooling")
            pooled_outputs.append(max_pooling)

    # 统计所有的filter
    total_filters = num_filters * len(filters_size)
    total_pool = tf.concat(pooled_outputs, 3)
    flattend_pool = tf.reshape(total_pool, (-1, total_filters))

    # dropout
    with tf.name_scope("dropout"):
        dropout = tf.nn.dropout(flattend_pool, KEEP_PROB)

    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=(total_filters, 1), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(1), name="b")

        logits = tf.add(tf.matmul(dropout, W), b)
        predictions = tf.nn.sigmoid(logits, name="predictions")

    # loss
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
        loss = loss + L2_LAMBDA * tf.nn.l2_loss(W)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # evaluation
    with tf.name_scope("evaluation"):
        correct_preds = tf.equal(tf.cast(tf.greater(predictions, 0.5), tf.float32), targets)
        accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=1))

# 训练模型
# 存储准确率
cnn_train_accuracy = []
cnn_test_accuracy = []

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./graphs/cnn", tf.get_default_graph())
    n_batches = int(x_train.shape[0] / BATCH_SIZE)

    for epoch in range(EPOCHES):
        total_loss = 0
        for x_batch, y_batch in get_batch(x_train, y_train):
            _, l = sess.run([optimizer, loss],
                            feed_dict={inputs: x_batch,
                                       targets: y_batch})
            total_loss += l

        train_corrects = sess.run(accuracy, feed_dict={inputs: x_train, targets: y_train})
        train_acc = train_corrects / x_train.shape[0]
        cnn_train_accuracy.append(train_acc)

        test_corrects = sess.run(accuracy, feed_dict={inputs: x_test, targets: y_test})
        test_acc = test_corrects / x_test.shape[0]
        cnn_test_accuracy.append(test_acc)

        print(
            "Training epoch: {}, Training loss: {:.4f}, Train accuracy: {:.4f}, Test accuracy: {:.4f}".format(epoch + 1,
                                                                                                              total_loss / n_batches,
                                                                                                              train_acc,
                                                                                                              test_acc))

    saver.save(sess, "checkpoints/cnn")
    writer.close()

plt.plot(cnn_train_accuracy)
plt.plot(cnn_test_accuracy)
plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of CNN model")
plt.legend(["train", "test"])

# 预测模型
# 在test上的准确率
with tf.Session() as sess:
    saver.restore(sess, "checkpoints/cnn")

    total_correct = sess.run(accuracy,
                             feed_dict={inputs: x_test, targets: y_test})

    print("The CNN model accuracy on test set: {:.2f}%".format(100 * total_correct / x_test.shape[0]))

# 在命令行执行tensorboard --logdir="./graphs/rnn" --port 6006可以看到模型的tensorboard




###########################################################################################
###########################################################################################
###########################################################################################
# CNN multi-channel

# 构建模型图
tf.reset_default_graph()

# 我在这里定义了5种filter，每种100个
filters_size = [2, 3, 4, 5, 6]
num_filters = 100
# 超参数
BATCH_SIZE = 256
EPOCHES = 8
LEARNING_RATE = 0.001
L2_LAMBDA = 10
KEEP_PROB = 0.8

with tf.name_scope("cnn_multichannels"):
    with tf.name_scope("placeholders"):
        inputs = tf.placeholder(dtype=tf.int32, shape=(None, 20), name="inputs")
        targets = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")
    # embeddings
    with tf.name_scope("embeddings"):
        # static embeddings
        static_embedding_matrix = tf.Variable(initial_value=static_embeddings,
                                              trainable=False,
                                              name="static_embedding_matrix")
        static_embed = tf.nn.embedding_lookup(static_embedding_matrix, inputs, name="static_embed")
        static_embed_expanded = tf.expand_dims(static_embed, -1, name="static_embed_expand")

        # non-static embeddings
        dynamic_embedding_matrix = tf.Variable(tf.random_normal(shape=(VOCAB_SIZE, EMBEDDING_SIZE), stddev=0.1),
                                               trainable=True,
                                               name="dynamic_embedding_matrix")
        dynamic_embed = tf.nn.embedding_lookup(dynamic_embedding_matrix, inputs, name="dynamic_embed")
        dynamic_embed_expanded = tf.expand_dims(dynamic_embed, -1, name="dynamic_embed_expand")

        # stack
        embed_expanded = tf.concat((static_embed_expanded, dynamic_embed_expanded), axis=-1, name="embed_expanded")

    pooled_outputs = []

    # 迭代多个filter
    for i, filter_size in enumerate(filters_size):
        with tf.name_scope("conv_maxpool_%s" % filter_size):
            # 注意这里filter的channel要指定为2
            filter_shape = [filter_size, EMBEDDING_SIZE, 2, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, mean=0.0, stddev=0.1), name="W")
            b = tf.Variable(tf.zeros(num_filters), name="b")

            conv = tf.nn.conv2d(input=embed_expanded,
                                filter=W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")

            # 激活
            a = tf.nn.relu(tf.nn.bias_add(conv, b), name="activations")
            # 池化
            max_pooling = tf.nn.max_pool(value=a,
                                         ksize=[1, SENTENCE_LIMIT_SIZE - filter_size + 1, 1, 1],
                                         strides=[1, 1, 1, 1],
                                         padding="VALID",
                                         name="max_pooling")
            pooled_outputs.append(max_pooling)

    total_filters = num_filters * len(filters_size)
    total_pool = tf.concat(pooled_outputs, 3)
    flattend_pool = tf.reshape(total_pool, (-1, total_filters))

    with tf.name_scope("dropout"):
        dropout = tf.nn.dropout(flattend_pool, KEEP_PROB)

    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=(total_filters, 1), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(1), name="b")

        logits = tf.add(tf.matmul(dropout, W), b)
        predictions = tf.nn.sigmoid(logits, name="predictions")

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
        loss = loss + L2_LAMBDA * tf.nn.l2_loss(W)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    with tf.name_scope("evaluation"):
        correct_preds = tf.equal(tf.cast(tf.greater(predictions, 0.5), tf.float32), targets)
        accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=1))

# 训练模型
multi_cnn_train_accuracy = []
multi_cnn_test_accuracy = []

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./graphs/multi_cnn", tf.get_default_graph())
    n_batches = int(x_train.shape[0] / BATCH_SIZE)

    for epoch in range(EPOCHES):
        total_loss = 0
        for x_batch, y_batch in get_batch(x_train, y_train):
            _, l = sess.run([optimizer, loss],
                            feed_dict={inputs: x_batch,
                                       targets: y_batch})
            total_loss += l

        train_corrects = sess.run(accuracy, feed_dict={inputs: x_train, targets: y_train})
        train_acc = train_corrects / x_train.shape[0]
        multi_cnn_train_accuracy.append(train_acc)

        test_corrects = sess.run(accuracy, feed_dict={inputs: x_test, targets: y_test})
        test_acc = test_corrects / x_test.shape[0]
        multi_cnn_test_accuracy.append(test_acc)

        print(
            "Training epoch: {}, Training loss: {:.4f}, Train accuracy: {:.4f}, Test accuracy: {:.4f}".format(epoch + 1,
                                                                                                              total_loss / n_batches,
                                                                                                              train_acc,
                                                                                                              test_acc))

    saver.save(sess, "checkpoints/multi_cnn")
    writer.close()

plt.plot(multi_cnn_train_accuracy)
plt.plot(multi_cnn_test_accuracy)
plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of multi-channel CNN model")
plt.legend(["train", "test"])

# 预测模型

# 在test上的准确率
with tf.Session() as sess:
    saver.restore(sess, "checkpoints/multi_cnn")

    total_correct = sess.run(accuracy,
                             feed_dict={inputs: x_test, targets: y_test})

    print("The MULTI-CNN model accuracy on test set: {:.2f}%".format(100 * total_correct / x_test.shape[0]))
