import numpy as np
import tensorflow as tf
import time

from sklearn import metrics
from tensorflow import dynamic_embedding as de
from tensorflow.keras.layers import Dense


class DNNModel(tf.keras.Model):

  def __init__(self):
    super(DNNModel, self).__init__()
    self.batch_size = 512
    self.dense_num = 13
    self.sparse_num = 26
    self.feature_num = self.dense_num + self.sparse_num
    self.embedding_size = 8
    self.d0 = Dense(
        400,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.d1 = Dense(
        400,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.d2 = Dense(
        1,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))

    self.deep_embeddings = de.get_variable(
        name="dynamic_embeddings",
        dim=8,
        initializer=tf.keras.initializers.RandomNormal(0.0, 0.005))

  def read_batch(self, data_file):
    '''
    Dataset source of criteo:
    for official format: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
    The train dataset has some bad record that only has 38 features.
    '''
    dataset = tf.data.TextLineDataset([data_file]). \
      prefetch(tf.data.experimental.AUTOTUNE). \
      batch(self.batch_size, drop_remainder=True). \
      map(self.parser, num_parallel_calls=1)
    return dataset

  def parser(self, record_batch):

    def _to_tensors(record_batch):
      '''This Dataset parser is for Criteo official format.'''
      decode_batch = [x.decode('utf-8') for x in record_batch]
      str_data = np.char.split(decode_batch, sep='\t')

      lb = np.vectorize(lambda x: x[0])(str_data).astype(np.float32).reshape(
          (-1, 1))

      ft_dense = np.asarray([x[1:14] for x in str_data]).reshape([-1])
      ft_dense = np.where(ft_dense == '', '0', ft_dense)
      ft_dense = ft_dense.astype(dtype=np.int64).reshape(
          [self.batch_size, self.dense_num])
      ft_dense = ft_dense + np.array(
          [(i + 1) * 0xFFFFFFFF for i in range(self.dense_num)])
      ft_dense = ft_dense.astype(dtype=np.int64).reshape(
          [self.batch_size, self.dense_num])

      ft_sparse = np.asarray([x[14:] for x in str_data]).reshape([-1])
      ft_sparse = np.where(ft_sparse == '', '0xFFFFFFFF', ft_sparse)
      ft_sparse = np.asarray([int(x, 16) for x in ft_sparse],
                             dtype=np.int64).reshape([-1, self.sparse_num])
      ft_sparse = np.concatenate([ft_dense, ft_sparse], axis=1)

      ft_sparse_val, ft_sparse_idx = np.unique(ft_sparse, return_inverse=True)
      ft_sparse_val = ft_sparse_val.astype(dtype=np.int64).reshape([-1])
      ft_sparse_idx = ft_sparse_idx.astype(dtype=np.int32).reshape([-1])

      return lb, ft_sparse_val, ft_sparse_idx

    ops = tf.numpy_function(_to_tensors,
                            inp=[record_batch],
                            Tout=[tf.float32, tf.int64, tf.int32])
    return ops

  def call(self, batch):
    labels, ft_sparse_val, ft_sparse_idx = batch
    deep_sparse_weights, self.trainable_wrappper = de.embedding_lookup(
        params=self.deep_embeddings,
        ids=ft_sparse_val,
        name="wide-sparse-weights",
        return_trainable=True)
    deep_embedding = tf.gather(deep_sparse_weights, ft_sparse_idx)
    deep_embedding = tf.reshape(
        deep_embedding,
        shape=[self.batch_size, self.feature_num * self.embedding_size])
    dnn = self.d0(deep_embedding)
    dnn = self.d1(dnn)
    logits = self.d2(dnn)
    predict = tf.nn.sigmoid(logits)
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(logits,
                                                                  shape=[-1]),
                                                labels=tf.reshape(labels,
                                                                  shape=[-1])))
    size = self.deep_embeddings.size()
    return labels, logits, predict, loss, size


dnn_model = DNNModel()
optimizer = tf.keras.optimizers.Adam(0.00025)


def train_step(batch):
  with tf.GradientTape() as tape:
    labels, logits, predict, loss, size = dnn_model(batch)
    tape.watch(dnn_model.trainable_wrappper)
  grads = tape.gradient(loss, dnn_model.trainable_variables)
  optimizer.apply_gradients(zip(grads, dnn_model.trainable_variables))
  return labels, predict, size


def train(epochs=1):
  dataset = dnn_model.read_batch("./criteo/train-512000.txt")
  for epoch in range(epochs):
    lb, pred = np.array([]), np.array([])
    for (tstep, batch) in enumerate(dataset):
      step_start = time.time()
      _lb, _pred, size = train_step(batch)
      step_end = time.time()
      if 100 - tstep <= (tstep % 100) < 100:
        lb = np.concatenate([lb, np.reshape(_lb, [-1])], 0)
        pred = np.concatenate([pred, np.reshape(_pred, [-1])], 0)
      if tstep % 100 == 0 and tstep > 0:
        auc = metrics.roc_auc_score(np.asarray(lb), np.asarray(pred))
        loss = metrics.log_loss(np.asarray(lb), np.asarray(pred))
        print("step{}:\tauc={:.4f}\tloss={:.4f}\tspd={:.4f}s/step\t size={}".
              format(tstep, float(auc), float(loss),
                     float(step_end - step_start), size))
    print('Epoch {} finished'.format(epoch))


if __name__ == "__main__":
  train()
