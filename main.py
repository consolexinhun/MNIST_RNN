import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, datasets, optimizers, Sequential, metrics, Model

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) /255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

batch_size = 128
lr = 1e-3
epochs = 20
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.map(preprocess).shuffle(1000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batch_size)

model=Sequential([
    # layers.Flatten(),
    # layers.Dense(128, activation='relu'),
    # layers.Dense(10)
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(64),
    layers.Flatten(),
    layers.Dense(10)
])

optimizer = optimizers.Adam(lr=lr)

for epoch in range(epochs):
    for step, (x, y) in enumerate(db_train):
        # x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:
            logits = model(x)
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))

    totol_correct = 0
    totol_num = 0
    for x, y in db_test:
        # x = tf.reshape(x, [-1, 28 * 28])
        logits = model(x)
        pred = tf.argmax(logits, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))

        totol_correct += int(correct)
        totol_num += x.shape[0]

    acc = totol_correct / totol_num
    print(epoch, 'test:', acc)
