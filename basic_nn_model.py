import tensorflow as tf
import pickle
import numpy as np

batch_size = 128
rnn_layers = 3

with open('output/train_preprocessed.csv', 'r') as f:
    columns = f.readline().strip().split(',')

with open('output/feature_dict.pickle', 'rb') as f:
    feature_dict = pickle.load(f)

print('columns:', columns)

x_len = len(columns) - 2

x = tf.placeholder(dtype=tf.int32, shape=(batch_size, x_len))
y = tf.placeholder(dtype=tf.int32, shape=(batch_size, 1))

x_embed = []

embeddings_list=[]

with tf.name_scope('embeddings'):
    embeddings = tf.Variable(
        tf.random_uniform([24, 3], -1.0, 1.0))
    embeddings_list.append(embeddings)
    embed = tf.nn.embedding_lookup(embeddings, x[:, 0])
    x_embed.append(embed)

for i in range(2, len(columns) - 1):
    with tf.name_scope('embeddings'):
        feature_size = len(feature_dict[columns[i]])
        embeddings = tf.Variable(
            tf.random_uniform([feature_size, 3], -1.0, 1.0))
        embeddings_list.append(embeddings)
        embed = tf.nn.embedding_lookup(embeddings, x[:, i - 1])
        x_embed.append(embed)

with tf.name_scope('embeddings'):
    embeddings = tf.Variable(
        tf.random_uniform([7, 3], -1.0, 1.0))
    embeddings_list.append(embeddings)
    embed = tf.nn.embedding_lookup(embeddings, x[:, x_len - 1])
    x_embed.append(embed)

x_embed = tf.concat(x_embed, 1)
x_embed_size = x_embed.shape.as_list()
w1 = tf.get_variable('w1', initializer=tf.truncated_normal((x_embed_size[1], x_embed_size[1])))
b1 = tf.get_variable('b1', initializer=tf.truncated_normal((1, x_embed_size[1])))
logists1 = tf.matmul(x_embed, w1) + b1
activate1 = tf.nn.tanh(logists1)

w2 = tf.get_variable('w2', initializer=tf.truncated_normal((x_embed_size[1], 1)))
b2 = tf.get_variable('b2', initializer=tf.truncated_normal((1, 1)))
logists2 = tf.matmul(activate1, w2) + b2
y_ = tf.nn.sigmoid(logists2)

loss = tf.losses.log_loss(y, y_)
tf.summary.scalar('loss', loss)

train_op = tf.train.AdamOptimizer(0.001)
train_op = train_op.minimize(loss=loss, global_step=tf.train.get_global_step())


sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

with open('output/train_preprocessed.csv', 'r') as f:
    f.readline()
    k = 0
    x_batch = []
    y_batch = []
    for line in f:
        k += 1
        tmp = list(map(lambda x: int(x), line.strip().split(',')))
        x_batch.append(tmp[2:])
        y_batch.append([tmp[1]])
        if k == batch_size:
            feed_dict = {
                x: np.asarray(x_batch),
                y: np.asarray(y_batch)
            }
            curr_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)

            k = 0
            x_batch = []
            y_batch = []
