import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import *

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) /255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.mnist.load_data()
db = tf.data.Dataset.from_tensor_slices((x, y))
batch_size = 128
db = db.map(preprocess).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batch_size)


model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10),
])
model.build(input_shape=(None, 28*28))
model.summary()

optimizer = optimizers.Adam(lr=1e-3)

def main():

    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                logits = model(x)

                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                # loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        totol_correct = 0
        totol_num = 0
        for x, y in db_test:
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))

            totol_correct += int(correct)
            totol_num += x.shape[0]

        acc = totol_correct / totol_num
        print(epoch, 'test:', acc)

if __name__ == '__main__':
    main()