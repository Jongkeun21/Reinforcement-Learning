import tensorflow as tf
import numpy as np
import gym

tf.logging.set_verbosity(tf.logging.ERROR)

env = gym.make('CartPole-v0')

x = tf.placeholder(shape=[1,4], dtype=tf.float32)
y = tf.placeholder(shape=[1,env.action_space.n], dtype=tf.float32)
learning_rate = tf.placeholder(dtype=tf.float32)

INPUT = env.observation_space.shape[0]
XAVIER = tf.contrib.layers.xavier_initializer()

W1 = tf.get_variable(shape=[INPUT,50], initializer=XAVIER, name='W1')
W2 = tf.get_variable(shape=[50,100], initializer=XAVIER, name='W2')
W3 = tf.get_variable(shape=[100,100], initializer=XAVIER, name='W3')
W4 = tf.get_variable(shape=[100,env.action_space.n], dtype=tf.float32, name='W4')

L1 = tf.nn.relu(tf.matmul(x,W1))
L2 = tf.nn.relu(tf.matmul(L1,W2))
L3 = tf.nn.relu(tf.matmul(L2,W3))
Q_ = tf.matmul(L3,W4)

initial_rate = 1e-2
num_episodes = 300
e = 0.01
df = 0.99
rList = []

loss = tf.reduce_sum(tf.square(y-Q_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init)

    for i in range(num_episodes) :
        s = env.reset()
        e = 1./((i/25) + 10)
        rAll = 0
        j = 0
        d = False
        s_t = sess.run(tf.expand_dims(s, axis=0))

        if i > 0 and i%100 == 0 :
            initial_rate *= 0.1
            print("Learning rate changed: ", initial_rate)

        while not d :
            j += 1
            Q = sess.run(Q_, feed_dict={x: s_t, learning_rate: initial_rate})

            if e > np.random.rand(1) :
                a = env.action_space.sample()
            else :
                a = np.argmax(Q)

            s1, r, d, _ = env.step(a)

            if d :
                Q[0,a] = -100
            else :
                s1_t = sess.run(tf.expand_dims(s1, axis=0))
                Q1 = sess.run(Q_, feed_dict={x: s1_t, learning_rate: initial_rate})

                Q[0,a] = r + (df*np.argmax(Q1))

            _ = sess.run(train, feed_dict={x: s_t, y: Q, learning_rate: initial_rate})

            rAll += r
            s_t = s1_t

        rList.append(rAll)
        print("Episode {} finished after {} timesteps with reward {}. Score: {}".format(i, j, rAll, np.mean(rList)))