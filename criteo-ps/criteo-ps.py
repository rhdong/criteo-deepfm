import os, time, sys
import tensorflow as tf
import numpy as np
from sklearn import metrics
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import cluster_pb2

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'ps_list', "localhost:2220, localhost:2221",
    'ps_list: to be a comma seperated string, '
    'like "localhost:2220, localhost:2220"')
flags.DEFINE_string(
    'worker_list', "localhost:2230",
    'worker_list: to be a comma seperated string, '
    'like "localhost:2230, localhost:2231"')
flags.DEFINE_string('task_mode', "worker", 'runninig_mode: ps or worker.')
flags.DEFINE_integer('task_id', 0, 'task_id: used for allocating samples.')
flags.DEFINE_bool('is_chief', False, ''
                  ': If true, will run init_op and save/restore.')


class Trainer(object):

  def __init__(self, worker_id, worker_num):
    self.epoch = 1
    self.batch_size = 512
    self.train_file = ["./criteo/train-512000.txt"]
    self.dense_num = 13
    self.sparse_num = 26
    self.feature_num = self.dense_num + self.sparse_num
    self.embedding_size = 10
    self.test_step = 16
    self.train_ratio = 0.8
    self.dense_weights = []
    self.cache_dataset = False
    self.add_wide = True
    self.add_deep = True
    self.add_fm = True
    self.profiler = False
    self.worker_id = worker_id
    self.ckpt_dir = "./ckpt/"

  def read_batch(self, data_file):
    '''
    Dataset source of criteo:
    for official format: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
    The train dataset has some bad record that only has 38 features.
    '''
    os.makedirs("./cache/", exist_ok=True)
    dataset = tf.data.TextLineDataset(data_file). \
      repeat(self.epoch). \
      batch(self.batch_size, drop_remainder=True). \
      map(self.parser, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
      prefetch(tf.data.experimental.AUTOTUNE)
    if self.cache_dataset:
      dataset = dataset.cache("./cache/")
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    return iterator

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

    ops = tf.compat.v1.py_func(_to_tensors,
                               stateful=False,
                               inp=[record_batch],
                               Tout=[tf.float32, tf.int64, tf.int32])
    return ops

  def fc(self, inputs, w_shape, b_shape, name):
    weight = tf.compat.v1.get_variable(name="%s_weights" % name,
                                       initializer=tf.random.normal(w_shape,
                                                                    mean=0.0,
                                                                    stddev=0.1))
    bias = tf.compat.v1.get_variable(name="%s_bias" % name,
                                     initializer=tf.random.normal(b_shape,
                                                                  mean=0.0,
                                                                  stddev=0.1))
    self.dense_weights.append(weight)
    self.dense_weights.append(bias)

    return tf.compat.v1.nn.xw_plus_b(inputs, weight, bias)

  def build_graph(self, data, global_step):
    labels, ft_sparse_val, ft_sparse_idx = data
    tf.compat.v1.set_random_seed(2020)
    labels = tf.reshape(labels, shape=[-1, 1])
    ft_sparse_val = tf.reshape(ft_sparse_val, shape=[-1])
    ft_sparse_idx = tf.reshape(ft_sparse_idx, shape=[-1])

    if self.add_wide:
      wide_dynamic_embeddings = tf.dynamic_embedding.get_variable(
          name="wide_dynamic_embeddings",
          devices=[
              "/job:ps/replica:0/task:0/CPU:0", "/job:ps/replica:0/task:1/CPU:0"
          ],
          initializer=tf.compat.v1.random_normal_initializer(0, 0.005),
          dim=1)
      wide_sparse_weights = tf.dynamic_embedding.embedding_lookup(
          params=wide_dynamic_embeddings,
          ids=ft_sparse_val,
          name="wide-sparse-weights")
      wide_embedding = tf.gather(wide_sparse_weights, ft_sparse_idx)

      wide_embedding = tf.reshape(wide_embedding,
                                  shape=[self.batch_size, self.feature_num])
      wide_bias = tf.compat.v1.get_variable(name="wide_bias",
                                            initializer=tf.random.normal(
                                                [1], mean=0.0, stddev=0.1))
      self.dense_weights.append(wide_bias)
      wide = tf.reshape(
          tf.reshape(tf.reduce_sum(input_tensor=wide_embedding, axis=1),
                     shape=[self.batch_size, 1]) + wide_bias,
          shape=[self.batch_size, 1])

    if self.add_deep:
      deep_dynamic_variables = tf.dynamic_embedding.get_variable(
          name="deep_dynamic_embeddings",
          devices=[
              "/job:ps/replica:0/task:0/CPU:0", "/job:ps/replica:0/task:1/CPU:0"
          ],
          initializer=tf.compat.v1.random_normal_initializer(0, 0.005),
          dim=self.embedding_size)
      deep_sparse_weights = tf.dynamic_embedding.embedding_lookup(
          params=deep_dynamic_variables,
          ids=ft_sparse_val,
          name="deep_sparse_weights")
      deep_embedding = tf.gather(deep_sparse_weights, ft_sparse_idx)
      deep_embedding = tf.reshape(
          deep_embedding,
          shape=[self.batch_size, self.feature_num * self.embedding_size])
      deep_embedding = tf.reshape([deep_embedding], shape=[self.batch_size, -1])
      deep = self.fc(deep_embedding,
                     [self.feature_num * self.embedding_size, 400], [400],
                     "fc_0")
      deep = tf.nn.dropout(deep, rate=0.2)
      deep = tf.nn.relu(deep)
      deep = self.fc(deep, [400, 400], [400], "fc_1")
      deep = tf.nn.dropout(deep, rate=0.2)
      deep = tf.nn.relu(deep)
      deep = self.fc(deep, [400, 1], [1], "fc_2")

    # # pair term
    if self.add_fm:
      x = tf.reshape(
          deep_embedding,
          shape=[self.batch_size, self.feature_num, self.embedding_size])
      fm = tf.reshape(0.5 * tf.reduce_sum(
          input_tensor=(tf.pow(tf.reduce_sum(input_tensor=x, axis=1), 2) -
                        tf.reduce_sum(input_tensor=tf.pow(x, 2), axis=[1])),
          axis=[1],
          keepdims=True),
                      shape=[self.batch_size, 1])

    logits = deep + wide + fm

    predict = tf.nn.sigmoid(logits)
    loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.reshape(logits, shape=[-1]),
        labels=tf.reshape(labels, shape=[-1])))
    opt = tf.compat.v1.train.AdamOptimizer(0.001 * self.batch_size / 2048)
    update = opt.minimize(loss, global_step=global_step)
    return {
        "update": update,
        "predict": predict,
        "label": labels,
        "size": deep_dynamic_variables.size(),
    }


def start_worker(worker_id, config):
  ps_list = config['cluster']['ps']
  worker_list = config['cluster']['localhost']
  worker_num = len(worker_list)

  # build cluster
  cluster_config = {
      'localhost': worker_list,
  }

  cluster_def = cluster_pb2.ClusterDef()
  worker_job = cluster_def.job.add()
  worker_job.name = 'worker'
  for i, v in enumerate(worker_list):
    worker_job.tasks[i] = v

  ps_job = cluster_def.job.add()
  ps_job.name = "ps"
  for i, v in enumerate(ps_list):
    ps_job.tasks[i] = v

  # build server
  sess_config = config_pb2.ConfigProto(
      cluster_def=cluster_def,
      experimental=config_pb2.ConfigProto.Experimental(
          share_session_state_in_clusterspec_propagation=True))

  sess_config.allow_soft_placement = False
  sess_config.log_device_placement = True
  cluster = tf.train.ClusterSpec(cluster_config)
  server = tf.distribute.Server(cluster,
                                protocol="grpc",
                                job_name="localhost",
                                task_index=worker_id,
                                config=sess_config)

  trainer = Trainer(worker_id, worker_num)
  with tf.device("/job:worker/replica:0/task:{}".format(worker_id)):
    train_iter = trainer.read_batch(trainer.train_file)
    train_data = train_iter.get_next()

  num_ps_tasks = len(ps_list)
  ps_stg = None  # tf.contrib.training.GreedyLoadBalancingStrategy(
  # num_ps_tasks, tf.contrib.training.byte_size_load_fn)
  device_setter = tf.compat.v1.train.replica_device_setter(
      ps_tasks=num_ps_tasks,
      worker_device="/job:worker",
      ps_device="/job:ps",
      ps_strategy=ps_stg)

  with tf.compat.v1.device(device_setter):
    outputs = trainer.build_graph(
        train_data, tf.compat.v1.train.get_or_create_global_step())

  if trainer.ckpt_dir:
    os.makedirs(os.path.split(trainer.ckpt_dir)[0], exist_ok=True)
  with tf.compat.v1.train.MonitoredTrainingSession(
      master=server.target,
      is_chief=FLAGS.is_chief,
      checkpoint_dir=trainer.ckpt_dir,
      config=sess_config,
      save_checkpoint_steps=400,
      log_step_count_steps=None,
  ) as sess:
    sess.run([train_iter.initializer])

    start_time = time.time()
    step, loss, auc = 0, 0, 0
    tstep = trainer.test_step
    lb, pred = np.array([]), np.array([])
    while True:
      step += 1
      step_start = time.time()
      _, _lb, _pred = sess.run(
          [outputs["update"], outputs["label"], outputs["predict"]])
      step_end = time.time()

      if 100 - tstep <= (step % 100) < 100:
        lb = np.concatenate([lb, _lb.reshape([-1])], 0)
        pred = np.concatenate([pred, _pred.reshape([-1])], 0)

      if step % 100 == 0:
        auc = metrics.roc_auc_score(np.asarray(lb), np.asarray(pred))
        loss = metrics.log_loss(np.asarray(lb), np.asarray(pred))
        _size = sess.run(outputs["size"])
        print("step{}:\tauc={:.4f}\tloss={:.4f}\tspd={:.4f}s/step\t size={}".
              format(step, float(auc), float(loss),
                     float(step_end - step_start), _size))
        lb, pred = np.array([]), np.array([])
      if step >= (0.8 * 2865038 / (trainer.batch_size * worker_num)):
        break

    print(
        "-" * 64, "\ntrain end\ttime={}\tfeatures size={}".format(
            time.time() - start_time, sess.run([outputs["size"]])))
    start_time = time.time()
    lb, pred = np.array([]), np.array([])
    try:
      step = 0
      while True:
        _lb, _pred = sess.run([outputs["label"], outputs["predict"]])
        lb = np.concatenate([lb, _lb.reshape([-1])], 0)
        pred = np.concatenate([pred, _pred.reshape([-1])], 0)
        step += 1

    except tf.errors.OutOfRangeError:
      auc = metrics.roc_auc_score(np.asarray(lb), np.asarray(pred))
      loss = metrics.log_loss(np.asarray(lb), np.asarray(pred))
      print(
          "-" * 64,
          "\ntest end\ttime={:.4f}/{}steps\tauc={:.4f}\tloss={:.4f}\n".format(
              time.time() - start_time, step, float(auc),
              float(loss)), "-" * 64)


def start_ps(task_id, config):
  cluster_def = cluster_pb2.ClusterDef()

  ps_job = cluster_def.job.add()
  ps_job.name = "ps"
  for i, v in enumerate(config["cluster"]["ps"]):
    ps_job.tasks[i] = v

  cluster = tf.train.ClusterSpec(config["cluster"])

  sess_config = tf.compat.v1.ConfigProto()
  sess_config.allow_soft_placement = False
  sess_config.log_device_placement = False
  server = tf.distribute.Server(cluster,
                                config=sess_config,
                                protocol='grpc',
                                job_name="ps",
                                task_index=task_id)
  server.join()


def main(argv):
  ps_list = FLAGS.ps_list.replace(' ', '').split(',')
  worker_list = FLAGS.worker_list.replace(' ', '').split(',')
  task_mode = FLAGS.task_mode
  task_id = FLAGS.task_id

  print('ps_list: ', ps_list)
  print('worker_list: ', worker_list)

  if task_mode == 'ps':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ps_config = {"cluster": {"ps": ps_list}}
    start_ps(task_id, ps_config)
  elif task_mode == 'worker':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(task_id)
    worker_config = {"cluster": {"ps": ps_list, "localhost": worker_list}}
    start_worker(task_id, worker_config)
  else:
    print('invalid task_mode. Options include "ps" and "worker".')
    sys.exit(1)


if __name__ == "__main__":
  tf.compat.v1.app.run()
