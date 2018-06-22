import tensorflow as tf
from tensorflow import logging
from tensorflow.python.client import device_lib
import tensorflow.contrib.slim as slim
from brand_entity_recm.models import BIRNNModel
from brand_entity_recm.batch import Dataset
import random

def get_gpus():
  local_device_protos = device_lib.list_local_devices()
  gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
  return gpus

def combine_gradients(tower_grads):
  """Calculate the combined gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been summed
     across all towers.
  """
  filtered_grads = [[x for x in grad_list if x[0] is not None] for grad_list in tower_grads]
  final_grads = []
  for i in range(len(filtered_grads[0])):
    grads = [filtered_grads[t][i] for t in range(len(filtered_grads))]
    grad = tf.stack([x[0] for x in grads], 0)
    grad = tf.reduce_sum(grad, 0)
    final_grads.append((grad, filtered_grads[0][i][1],))

  return final_grads



def build_graph(param_dict):
    print(param_dict)
    gpus = get_gpus()
    num_gpus = len(gpus)

    if num_gpus > 0:
        logging.info("Using the following GPUs to train: " + str(gpus))
        num_towers = num_gpus
        device_string = '/gpu:%d'
    else:
        logging.info("No GPUs found. Training on CPU.")
        num_towers = 1
        device_string = '/cpu:%d'

    global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer,
                                  trainable=False)

    optimizer = tf.train.AdamOptimizer(param_dict['learning_rate'])

    x = tf.placeholder(tf.int32, [None, param_dict['vec_length']], name="pl_tokens")
    y = tf.placeholder(tf.int32, [None, param_dict['vec_length']], name="pl_target")
    w = tf.placeholder(tf.float32, [None, param_dict['vec_length']], name="pl_weight")
    keep_prob = tf.placeholder(tf.float32, [], name="pl_keep_prob")

    tower_input_x = tf.split(x, num_towers)
    tower_input_y = tf.split(y, num_towers)
    tower_input_w = tf.split(w, num_towers)

    tower_loss = []
    tower_gradients = []
    tower_preds = []
    tower_probs = []

    model = BIRNNModel()

    for i in range(num_towers):
        with tf.device(device_string % i):
            with (tf.variable_scope("tower", reuse=True if i > 0 else None)):
                with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus!=1 else "/gpu:0")):
                    model_input = {'x': tower_input_x[i], 'y':tower_input_y[i], 'w': tower_input_w[i]}
                    step_outputs, loss = model.create_model(model_input=model_input, model_param=param_dict, keep_probs=keep_prob)

                    step_out_probs = []
                    step_out_preds = []
                    for _output in step_outputs:
                        _out_probs = tf.nn.softmax(_output)
                        _out_pred = tf.argmax(_out_probs, 1)

                        step_out_probs.append(_out_probs)
                        step_out_preds.append(_out_pred)

                    # stack for interface
                    step_out_probs = tf.stack(step_out_probs, axis=1, name="step_out_probs")
                    step_out_preds = tf.stack(step_out_preds, axis=1, name="step_out_preds")
                    tower_probs.append(step_out_probs)
                    tower_preds.append(step_out_preds)

                    tower_loss.append(loss)

                    gradients = optimizer.compute_gradients(loss)
                    tower_gradients.append(gradients)

    merged_loss = tf.reduce_mean(tf.stack(tower_loss))
    merged_gradients = combine_gradients(tower_gradients)

    train_op = optimizer.apply_gradients(merged_gradients, global_step=global_step)

    tf.add_to_collection('global_step', global_step)
    tf.add_to_collection('loss', merged_loss)
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('probs', tf.concat(tower_probs, 0))
    tf.add_to_collection('preds', tf.concat(tower_preds, 0))
    tf.add_to_collection('x_placeholder', x)
    tf.add_to_collection('y_placeholder', y)
    tf.add_to_collection('w_placeholder', w)
    tf.add_to_collection('keep_probs',keep_prob)

def train():
    param = {
        'vec_length':128,
        'enc_dim': 50,
        'batch_size':1000,
        'epochs':2,
        'emb_size':50,
        'learning_rate':0.01,
        'num_target_class':3
    }
    path = 'data/test_data_set.csv'
    dataset = Dataset(datapath=path, param=param)

    param['vocab_size'] = len(dataset.vocab)

    print(param)
    build_graph(param)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    global_step = tf.get_collection('global_step')[0]
    loss = tf.get_collection('loss')[0]
    train_op = tf.get_collection('train_op')[0]
    probs = tf.get_collection('probs')[0]
    preds = tf.get_collection('preds')[0]
    x = tf.get_collection('x_placeholder')[0]
    y = tf.get_collection('y_placeholder')[0]
    w = tf.get_collection('w_placeholder')[0]
    keep_probs = tf.get_collection('keep_probs')[0]

    for epoch in range(param['epochs']):
        print(epoch, 'init_iterator')
        dataset.init_iter()
        while True:
            try:
                x_batch, y_batch, w_batch = next(dataset.iterator)
                _, global_step_val, loss_val = sess.run([train_op, global_step, loss], feed_dict={x:x_batch,y:y_batch, w:w_batch, keep_probs:0.5})
                print(epoch, global_step_val, loss_val)

                x_test, y_test, w_test, row = dataset.row2input(random.choice(range(50000)),1)
                test_case = sess.run(preds, feed_dict={x:x_test,y:y_test,w:w_test,keep_probs:1})
                print(row['proc_prdnm'].iloc[0], ' : ', row['proc_attr'].iloc[0])
                print(''.join(test_case[0].astype(str)))

            except StopIteration:
                break


if __name__ == '__main__':
    train()
