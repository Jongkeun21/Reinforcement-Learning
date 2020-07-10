import tensorflow as tf
import numpy as np
import gym
import random as ran

from gym import wrappers

env = gym.make('CartPole-v0')

REPLAY = 50
REPLAY_MEMORY = []
MINIBATCH = 50

INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n

RATE = 1e-2
DISCOUNTING = 0.99
PATH = "save/model.ckpt"
XAVIER = tf.contrib.layers.xavier_initializer()

x = tf.placeholder(shape=[None,INPUT], dtype=tf.float32)
y = tf.placeholder(shape=[None,OUTPUT], dtype=tf.float32)
dropout = tf.placeholder(dtype=tf.float32)

W1 = tf.get_variable(name='W1', shape=[INPUT,50], initializer=XAVIER)
W2 = tf.get_variable(name='W2', shape=[50,100], initializer=XAVIER)
W3 = tf.get_variable(name='W3', shape=[100,OUTPUT], initializer=XAVIER)

b1 = tf.Variable(tf.zeros([1], dtype=tf.float32))
b2 = tf.Variable(tf.zeros([1], dtype=tf.float32))

L1 = tf.nn.relu(tf.matmul(x,W1)+b1)
L1 = tf.nn.dropout(L1, dropout)

L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
L2 = tf.nn.dropout(L2, dropout)

Q_pre = tf.matmul(L2,W3)

W1_r = tf.get_variable(name='W1_r', shape=[INPUT,50])
W2_r = tf.get_variable(name='W2_r', shape=[50,100])
W3_r = tf.get_variable(name='W3_r', shape=[100,OUTPUT])

b1_r = tf.Variable(tf.zeros([1], dtype=tf.float32))
b2_r = tf.Variable(tf.zeros([1], dtype=tf.float32))

L1_r = tf.nn.relu(tf.matmul(x,W1_r)+b1_r)
L2_r = tf.nn.relu(tf.matmul(L1_r,W2_r)+b2_r)
Q_pre_r = tf.matmul(L2_r,W3_r)

rList = [0]
recentRList = [0]
episode = 0

loss = tf.reduce_sum(tf.square(y-Q_pre))
optimizer = tf.train.AdamOptimizer(learning_rate=RATE)
train = optimizer.minimize(loss)
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init)
    sess.run([W1_r.assign(W1), W2_r.assign(W2), W3_r.assign(W3), b1_r.assign(b1), b2_r.assign(b2)])

    while np.mean(recentRList) < 195 :
        episode += 1
        s = env.reset()

        if len(recentRList) > 200 :
            del recentRList[0]

        e = 1./((episode/25)+1)
        rAll = 0
        d = False
        j = 0

        while not d and j < 10000 :
            j += 1
            s_t = np.reshape(s, [1,INPUT])

            Q = sess.run(Q_pre, feed_dict={x: s_t, dropout: 0.3})

            if e > np.random.rand(1) :
                a = env.action_space.sample()
            else :
                a = np.argmax(Q)

            s1, r, d, _ = env.step(a)
            REPLAY_MEMORY.append([s_t, a, r, s1, d, j])

            if len(REPLAY_MEMORY) > 50000 :
                del REPLAY_MEMORY[0]

            rAll += r
            s = s1

        if len(REPLAY_MEMORY) > MINIBATCH:
            for sample in ran.sample(REPLAY_MEMORY, REPLAY) :
                s_t_r, a_r, r_r, s1_r, d_r, j_r = sample

                y_ = sess.run(Q_pre, feed_dict={x: s_t_r, dropout: 1})

                if d_r :
                    if j_r < env.spec.timestep_limit :
                        y_[0,a_r] = -100
                else :
                    s1_t_r = np.reshape(s1_r, [1,INPUT])
                    Q1, Q = sess.run([Q_pre_r, Q_pre], feed_dict={x: s1_t_r, dropout: 0.3})
                    y_[0,a_r] = r_r + DISCOUNTING*Q1[0,np.argmax(Q)]

                _, loss_ = sess.run([train, loss], feed_dict={x: s_t_r, y: y_, dropout: 0.3})

            sess.run([W1_r.assign(W1), W2_r.assign(W2), W3_r.assign(W3), b1_r.assign(b1), b2_r.assign(b2)])
            print("Loss: ", loss_)

        recentRList.append(rAll)
        rList.append(rAll)

        print("Episode: {}, steps: {}, reward: {}, average reward: {}, recent reward: {}".format(episode, j, rAll, np.mean(rList), np.mean(recentRList)))

    save_path = saver.save(sess, PATH)
    print("Model saved in file: ", save_path)

    rList = []
    recentRList = []