# Alfan Farizki Wicaksono
# Fasilkom, Universitas Indonesia
# simple Recursive Auto-Encoder

import tensorflow as tf
import numpy as np
import sys
#from sklearn import cluster, datasets, mixture


#
#        y2
#       /  \
#      y1   \
#     /  \   \
#    x1  x2   x3
#

# input: x1, x2, x3
# y1 = f(W, [x1;x2])
# y2 = f(W, [y1;x3])
#
# [x1';x2'] = f(W', y1)
# [y1';x3'] = f(W', y2)
#
# Loss = L([x1;x2], [x1';x2']) + L([y1;x3], [y1';x3'])
#

nsample = 70
nfeature = 300

# generate data ecek-ecek/pura-pura untuk x1, x2, x3
# x1,x2,x3 adalah tensor berukuran 10 x 5, panjang fitur untuk masing-masing xi adalah 5
#X1 = np.random.rand(nsample, nfeature)
#X2 = np.random.rand(nsample, nfeature)
#X3 = np.random.rand(nsample, nfeature)
X1 = np.load("x1_astretestsample_70.npy")
X2 = np.load("x2_astretestsample_70.npy")
X3 = np.load("x3_astretestsample_70.npy")

# dense layer/fully-connected layer
# x: input berdimensi [batch, input unit]
# output_unit: integer, banyaknya output unit
def dense(x, output_unit, activation_fn, scope_name='dense', reuse=False):

    input_unit = x.get_shape()[-1]
    bias_shape = [output_unit]

    with tf.variable_scope(scope_name) as scope:

        # Jika kita ingin reuse parameters; atau pakai parameter dengan nama yang sama
        # kalau tidak di-reuse, maka kita tidak boleh membuat 2 atau lebih parameter
        # dengan nama scope yang sama!
        if reuse:
            scope.reuse_variables()

        # Membuat sebuah variable untuk "dense/weights"; hindari penggunaan tf.Variable(.)!!
        weights = tf.get_variable("weights", \
                                  [input_unit, output_unit], \
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))

        # Membuat sebuah variable untuk "dense/biases"
        biases = tf.get_variable("biases", \
                                 bias_shape, \
                                 initializer=tf.constant_initializer(0.0))

        y = tf.matmul(x, weights) + biases # raw logits
        
        if activation_fn == None:
            return y
        return activation_fn(y)


# reconstruction error
def loss(yy, yy_out):
    #disini, cocoknya pakai mean-squared error -> reconstruction error
    mse = tf.reduce_mean(tf.square(yy - yy_out))
    return mse

# untuk setiap batch
# fungsi ini digunakan saat training untuk mengambil batch demi batch pada sample
def gen_batch(x1, x2, x3, batch_size=2):
    num_sample, vector_size = x1.shape

    i = 0
    while i <= num_sample - batch_size:
        yield x1[i:(i+batch_size),:], x2[i:(i+batch_size),:], x3[i:(i+batch_size),:]
        i = i + batch_size


#rakit
batch_size = 1

x1 = tf.placeholder(shape=[batch_size, nfeature], dtype=tf.float32)
x2 = tf.placeholder(shape=[batch_size, nfeature], dtype=tf.float32)
x3 = tf.placeholder(shape=[batch_size, nfeature], dtype=tf.float32)

#encoder
x1x2 = tf.concat([x1, x2], 1)
y1 = dense(x1x2, output_unit=nfeature, activation_fn=tf.nn.tanh, scope_name='W1_dense')
y1x3 = tf.concat([y1, x3], 1)
y2 = dense(y1x3, output_unit=nfeature, activation_fn=tf.nn.tanh, scope_name='W1_dense', reuse=True)

_x1x2 = dense(y1, output_unit=nfeature*2, activation_fn=None, scope_name='W2_dense')
_y1x3 = dense(y2, output_unit=nfeature*2, activation_fn=None, scope_name='W2_dense', reuse=True)

#decoder


LRAE = loss(x1x2, _x1x2) + loss(y1x3, _y1x3)


# optimizer
# pakai SGD, learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(0.05)
train_onestep = optimizer.minimize(LRAE) #mengembalikan None, ini untuk update parameter

# training
def train(sess, epoch=10000, step=10):
    init = tf.global_variables_initializer()

    #inisialisasi variables/parameters
    sess.run(init)

    #mulai epoch
    for i in range(epoch):

        #untuk setiap batch pada dataset
        for x, y, z in gen_batch(X1, X2, X3, batch_size=batch_size):
            #print("x1 shape di loop train= ")
            #print(str(x1.shape))
            #print("x shape di loop train= ")
            #print(str(x.shape))

            # INGAT! di autoencoder set input = output
            _,loss = sess.run([train_onestep, LRAE], feed_dict={x1:x, x2:y, x3:z})

            #cetak progress untuk setiap 'step' epoch
            if (i % step == 0):
                sys.stdout.write("\r Epoch-{}, Loss-value: {}".format(i+1, loss))
                sys.stdout.flush()

        if (i % step == 0):
            sys.stdout.write("\n")
            sys.stdout.flush()


with tf.Session() as sess:

    ### training ###
    train(sess)

    XX1 = tf.placeholder(shape=[nsample, nfeature], dtype=tf.float32)
    XX2 = tf.placeholder(shape=[nsample, nfeature], dtype=tf.float32)
    XX3 = tf.placeholder(shape=[nsample, nfeature], dtype=tf.float32)

    XX1XX2 = tf.concat([XX1, XX2], 1)
    YY1 = dense(XX1XX2, output_unit=nfeature, activation_fn=tf.nn.tanh, scope_name='W1_dense', reuse=True)
    YY1XX3 = tf.concat([YY1, XX3], 1)
    YY2 = dense(YY1XX3, output_unit=nfeature, activation_fn=tf.nn.tanh, scope_name='W1_dense', reuse=True)

    _XX1XX2 = dense(YY1, output_unit=nfeature*2, activation_fn=None, scope_name='W2_dense', reuse=True)
    _YY1XX3 = dense(YY2, output_unit=nfeature*2, activation_fn=None, scope_name='W2_dense', reuse=True)


    encoded_data = sess.run([YY2], feed_dict={XX1:X1, XX2:X2, XX3:X3})
    np.save("encoded_astretestsample_70_10000",encoded_data)

    print("data X1: ")
    print(str(X1))

    print("data X2: ")
    print(str(X2))

    print("data X3: ")
    print(str(X3))

    print("encoded data: ")
    print(str(encoded_data[0]))

    #coba decoder
    _Y1 = tf.placeholder(shape=[nsample, nfeature], dtype=tf.float32)
    _X3 = tf.placeholder(shape=[nsample, nfeature], dtype=tf.float32)
    _Y1_X3 = tf.placeholder(shape=[nsample, nfeature], dtype=tf.float32)
    _X2 = tf.placeholder(shape=[nsample, nfeature], dtype=tf.float32)
    _X1 = tf.placeholder(shape=[nsample, nfeature], dtype=tf.float32)
    _X1_X2 = tf.placeholder(shape=[nsample, nfeature], dtype=tf.float32)

    #decode tahap 1
    Y1X3 = dense(_Y1_X3, output_unit=nfeature*2, activation_fn=None, scope_name='W2_dense', reuse=True)
    decoded_data_1 = sess.run([Y1X3], feed_dict={_Y1_X3:encoded_data[0]})

    #split data Y1 dan X3
    decoded_data_1_part = np.array_split(decoded_data_1[0], 2, axis=1)
    print("decoded_data_1_part[0]:")
    print(decoded_data_1_part[0])
    print("X3:")
    print(decoded_data_1_part[1])

    #split tensor Y1 dan X3
    _Y1, _X3 = tf.split(Y1X3, 2, axis=1)

    #decode tahap 2
    X1X2 = dense(_Y1, output_unit=nfeature*2, activation_fn=None, scope_name='W2_dense', reuse=True)
    decoded_data_2 = sess.run([X1X2], feed_dict={_Y1:decoded_data_1_part[0]})

    #split data X1 dan X2
    decoded_data_2_part = np.array_split(decoded_data_2[0], 2, axis=1)
    print("X1:")
    print(decoded_data_2_part[0])
    print("X2:")
    print(decoded_data_2_part[1])    
