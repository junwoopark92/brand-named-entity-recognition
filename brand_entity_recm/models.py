import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.layers.python.layers import linear
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.seq2seq import sequence_loss


class BaseModel(object):
    def create_model(self, unused_model_input, unused_params):
        raise NotImplementedError()


class BIRNNModel(BaseModel):
    x = None
    y = None
    z = None

    def _embedding(self, model_input, **param):
        shape = [param['vocab_size'], param['emb_size']]
        initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True, dtype=tf.float32)
        emb_mat = tf.get_variable("emb", shape, initializer=initializer, dtype=tf.float32)
        print(emb_mat.shape)
        ## embedding added

        input_emb = tf.nn.embedding_lookup(emb_mat, model_input)  # [batch_size, sent_len, emb_dim]

        # split input_emb -> num_steps
        step_inputs = tf.unstack(input_emb, axis=1)
        return step_inputs

    def _sequence_dropout(self, step_inputs, keep_prob):
        # apply dropout to each input
        # input : a list of input tensor which shape is [None, input_dim]
        with tf.name_scope('sequence_dropout') as scope:
            step_outputs = []
            for t, input in enumerate(step_inputs):
                step_outputs.append(tf.nn.dropout(input, keep_prob))
        return step_outputs

    def sequence_encoding_n2n(self, step_inputs, seq_length, cell_size):
        # birnn based N2N encoding and output
        f_rnn_cell = tf.contrib.rnn.GRUCell(cell_size, reuse=False)
        b_rnn_cell = tf.contrib.rnn.GRUCell(cell_size, reuse=False)
        _inputs = tf.stack(step_inputs, axis=1)

        # step_inputs = a list of [batch_size, emb_dim]
        # input = [batch_size, num_step, emb_dim]
        # np.stack( [a,b,c,] )
        outputs, states, = tf.nn.bidirectional_dynamic_rnn(f_rnn_cell,
                                                           b_rnn_cell,
                                                           _inputs,
                                                           sequence_length=tf.cast(seq_length, tf.int64),
                                                           time_major=False,
                                                           dtype=tf.float32,
                                                           scope='birnn',
                                                           )
        output_fw, output_bw = outputs
        states_fw, states_bw = states

        output = tf.concat([output_fw, output_bw], 2)
        step_outputs = tf.unstack(output, axis=1)

        final_state = tf.concat([states_fw, states_bw], 1)
        return step_outputs  # a list of [batch_size, enc_dim]

    def _to_class_n2n(self, step_inputs, num_class):
            T = len(step_inputs)
            step_output_logits = []
            for t in range(T):
                # encoder to linear(map)
                out = step_inputs[t]
                if t==0: out = linear(out, num_class, scope="Rnn2Target")
                else:    out = linear(out, num_class, scope="Rnn2Target", reuse=True)
                step_output_logits.append(out)
            return step_output_logits

    def _loss(self, step_outputs, step_refs, weights):
        # step_outputs : a list of [batch_size, num_class] float32 - unscaled logits
        # step_refs    : [batch_size, num_steps] int32
        # weights      : [batch_size, num_steps] float32
        # calculate sequence wise loss function using cross-entropy
        _batch_output_logits = tf.stack(step_outputs, axis=1)
        loss = sequence_loss(
                                logits=_batch_output_logits,
                                targets=step_refs,
                                weights=weights
                            )
        return loss

    def create_model(self, model_input, model_param, **kwargs):
        print(model_input)

        seq_length = tf.reduce_sum(model_input['w'], 1)

        step_inputs = self._embedding(model_input['x'], vocab_size=model_param['vocab_size'], emb_size=model_param['emb_size'])
        step_inputs = self._sequence_dropout(step_inputs, kwargs['keep_probs'])
        step_enc_outputs = self.sequence_encoding_n2n(step_inputs, seq_length, model_param['enc_dim'])
        step_outputs = self._to_class_n2n(step_enc_outputs, model_param['num_target_class'])

        loss = self._loss(step_outputs, model_input['y'], model_input['w'])

        return step_outputs, loss

