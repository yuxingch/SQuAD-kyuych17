# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

from tensorflow.contrib import rnn


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        # use GRU
        #self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        #self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        #self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        #self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)
        # use LSTM
        self.lstm_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.lstm_cell_fw = DropoutWrapper(self.lstm_cell_fw, input_keep_prob=self.keep_prob)
        self.lstm_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.lstm_cell_bw = DropoutWrapper(self.lstm_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            #(fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell_fw, self.lstm_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)
            # Add the forward and backward hidden states
            #out = fw_out + bw_out 
            # Average the forward and backward hidden states
            #out = (fw_out + bw_out) / 2
            # Max pool the forward and backward hidden states
            #out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class RNNEncoderV2(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        # use GRU
        #self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        #self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        #self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        #self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)
        # use LSTM
        self.lstm_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.lstm_cell_fw = DropoutWrapper(self.lstm_cell_fw, input_keep_prob=self.keep_prob)
        self.lstm_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.lstm_cell_bw = DropoutWrapper(self.lstm_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            #(fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell_fw, self.lstm_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Apply dropout
            fw_out = tf.nn.dropout(fw_out, self.keep_prob)
            bw_out = tf.nn.dropout(bw_out, self.keep_prob)

            return fw_out, bw_out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class BidirectionalAttn(object):
    """Module for bidirectional attention flow.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          C2Q: Tensor shape (batch_size, num_keys, hidden_size).
            This is the C2Q attention output; the weighted sum of the values
            (using the attention distribution as weights).
          Q2C: Tensor shape (batch_size, num_keys, hidden_size).
            This is the Q2C attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BiDAF"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            masked_logits, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # C2Q attention
            # Use attention distribution to take weighted sum of values
            C2Q = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)
            # Apply dropout
            C2Q = tf.nn.dropout(C2Q, self.keep_prob)

            # Q2C attention
            m = tf.reduce_max(masked_logits, axis=2) # shape (batch_size, num_keys)
            _, beta = masked_softmax(m, keys_mask, 1) # shape (batch_size, num_keys)
            beta = tf.expand_dims(beta, axis=1) # shape (batch_size, 1, num_keys)
            Q2C = tf.matmul(beta, keys) # shape (batch_size, 1, value_vec_size)
            N = tf.shape(keys)[1]
            Q2C = tf.tile(Q2C, [1, N, 1]) # shape (batch_size, num_keys, value_vec_size)

            return C2Q, Q2C


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


class ModelingForBiDAF(object):
    """
    """

    def __init__(self, hidden_size, keep_prob, num_layers=3):
        """
        Inputs:
          hidden_size: int. Hidden size of the LSTM
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
          num_layers: int. Number of layers of the LSTM
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        
        # use LSTM
        lstm_cells_fw = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size)  
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
            lstm_cells_fw.append(cell)
        self.lstm_cell_fw = tf.contrib.rnn.MultiRNNCell(lstm_cells_fw)

        lstm_cells_bw = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size)  
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
            lstm_cells_bw.append(cell)
        self.lstm_cell_bw = tf.contrib.rnn.MultiRNNCell(lstm_cells_bw)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("ModelingForBiDAF"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell_fw, self.lstm_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class Multi_Highway(object):
    """
    """

    def __init__(self, hidden_size, keep_prob, num_layers=3):
        """
        Inputs:
          hidden_size: int. Hidden size of the LSTM
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
          num_layers: int. Number of layers of the LSTM
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        
    def build_graph(self, inputs):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        scope_name = 'highway_layer'
        for i in range(self.num_layers):
            cur_scope_name = scope_name + "-{}".format(i)
            inputs = self.highway_layer(inputs, self.hidden_size, scope=cur_scope_name)
        return inputs

    def highway_layer(self, inputs, output_size, scope=None):
        # inputs: (batch_size, context_len, input_size)
        B = tf.shape(inputs)[0]
        N = tf.shape(inputs)[1]
        B1 = inputs.shape.as_list()[0]
        N1 = inputs.shape.as_list()[1]
        d = inputs.shape.as_list()[2]
        inputs = tf.reshape(inputs, [B * N, d])
        with tf.variable_scope(scope):
            full_W = tf.get_variable("full_W", [d, output_size], dtype=tf.float32)
            full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
            
            highway_W = tf.get_variable("highway_W", [d, output_size], dtype=tf.float32)
            highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
            
            trans = tf.nn.tanh(tf.nn.xw_plus_b(inputs, full_W, full_b))
            gate = tf.nn.sigmoid(tf.nn.xw_plus_b(inputs, highway_W, highway_b))
            output = trans*gate + inputs * (1.0 - gate)
            output = tf.reshape(output, [B, N, output_size])
            output = tf.nn.dropout(output, self.keep_prob)
        B = inputs.shape.as_list()[0]
        N = inputs.shape.as_list()[1]
        output.set_shape([B1, N1, output_size])
        #print output.get_shape()
	return output


class GatedAttn(object):
    """
    Module for gated attention.
    """

    def __init__(self, keep_prob):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.keep_prob = keep_prob

    def build_graph(self, states_Q, states_P, Q_mask):
        """
        

        Inputs:
          states_Q: Tensor shape (batch_size, question_size, vec_size)
          states_P: Tensor shape (batch_size, passage_size, vec_size)
          Q_mask: Tensor shape (batch_size, question_size).
            1s where there's real input, 0s where there's padding

        Outputs:
          states_bm: Tensor shape (batch_size, question_size, 2*vec_size).
        """
        with vs.variable_scope("GatedAttn"):

            # states_Q: Tensor shape (batch_size, question_size, vec_size)
            # states_P: Tensor shape (batch_size, passage_size, vec_size)
            B = states_Q.shape.as_list()[0] # batch size
            M = states_Q.shape.as_list()[1] # question size
            N = states_P.shape.as_list()[1] # passage size, number of timesteps
            d = states_Q.shape.as_list()[2] # vec size

            # Define attention parameters
            W_u_Q = tf.get_variable('W_u_Q', shape=[d, d], dtype=tf.float32)
            W_u_P = tf.get_variable('W_u_P', shape=[d, d], dtype=tf.float32)
            W_v_P = tf.get_variable('W_v_Q', shape=[d, d], dtype=tf.float32)
            V = tf.get_variable('V', shape=[d, 1], dtype=tf.float32)

            attn_params = {'W_u_Q': W_u_Q, 'W_u_P': W_u_P, 'W_v_P': W_v_P, 'V': V}

            W_g = tf.get_variable('W_g', shape=[d, d], dtype=tf.float32) # gate 

            # Define attention cell 
            cell = rnn.GRUCell(num_units = 2*d)

            # batch major -> timestep major
            states_P = tf.transpose(states_P, [1,0,2]) # (N, B, d)

            initial_state = tf.zeros(dtype=tf.float32, shape=[50, d])
            states = [initial_state]

            for t in range(N): # N timesteps (passage size)
                if t > 0:
                    tf.get_variable_scope().reuse_variables()

                ct = self.attention_pooling(states_Q, states_P[t], states[-1], attn_params, Q_mask, d, M) # [B, d]

                #inputs = tf.concat([states_P[t], ct], axis = -1) # [B, 2*d]
                inputs = ct

                # Define gate
                gate = tf.nn.sigmoid(tf.matmul(inputs, W_g)) # [B, 2*d]

                # Apply gate 
                inputs = gate * inputs # [B, 2*d]

                _, state = cell(inputs, states[-1]) # [B, 2*d]

                states.append(state)

            # timestep major -> batch major
            states_bm = tf.transpose(tf.stack(states[1:]), [1, 0, 2]) # [B, N, 2*d]

            return states_bm


    def attention_pooling(self, states_Q, states_P_t, prev_v, params, mask, d, M):
        # states_Q: Tensor shape (B, M, d)
        # states_P_t: Tensor shape (B, d)
        # prev_v: Tensor shape (B, 2*d)
        # mask: Tensor shape (B, M)
        # W_u_P [d, d], # W_u_Q [d, d], # W_v_P [2*d, d]
        W_u_P, W_u_Q, W_v_P = params['W_u_P'], params['W_u_Q'], params['W_v_P']
        z = tf.reshape(tf.matmul(tf.reshape(states_Q, [-1, d]), W_u_P), [-1, M, d]) # [B, M, d]
        z += tf.expand_dims(tf.matmul(states_P_t, W_u_Q), axis=1) # [B, M, d]
        z += tf.expand_dims(tf.matmul(prev_v, W_v_P), axis=1) # [B, M, d]
        a = tf.tanh(z) # [B, M, d]
        V = params['V'] # [d, 1]
        s = tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), V), [-1, M]) # [B, M]
        
        # Masking
        exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
        masked_s = tf.add(s, exp_mask) # where there's padding, set logits to -large
        # The distribution should sum to 1, and should be 0 in the value locations that correspond to padding.
        dist = tf.nn.softmax(masked_s) # [B, M] 
        
        # Use the attention distribution as weights
        ct = tf.reduce_sum(states_Q * tf.expand_dims(dist, axis=-1), axis=1) # [B, d]
        
        return ct


class CoAttn(object):
    """Module for Coattention.

    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size 
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.lstm_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.lstm_cell_fw = DropoutWrapper(self.lstm_cell_fw, input_keep_prob=self.keep_prob)
        self.lstm_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.lstm_cell_bw = DropoutWrapper(self.lstm_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, states_Q, states_P, Q_mask, P_mask):
        """
    

        Inputs:
          states_Q: Tensor shape (batch_size, question_size, vec_size)
          states_P: Tensor shape (batch_size, passage_size, vec_size)
          Q_mask: Tensor shape (batch_size, question_size).
          P_mask: Tensor shape (batch_size, passage_size).
            1s where there's real input, 0s where there's padding

        Outputs:
          out: Tensor shape (, , ).
        """
        with vs.variable_scope("CoAttn"):

            M = states_Q.shape.as_list()[1] # question size
            N = states_P.shape.as_list()[1] # passage size, number of timesteps
            d = states_Q.shape.as_list()[2] # vec size

            # Define attention parameters
            W_c = tf.get_variable('W_c', shape=[d, d], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            b_c = tf.get_variable('b_c', shape=[d, ], dtype=tf.float32, initializer=tf.constant_initializer(0))

            sentinel_P = tf.get_variable('s_P', states_P.get_shape()[2], dtype=tf.float32)
            sentinel_P = tf.reshape(sentinel_P, (1, 1, -1))
            sentinel_P = tf.tile(sentinel_P, (tf.shape(states_P)[0], 1, 1))

            sentinel_Q = tf.get_variable('s_Q', states_Q.get_shape()[2], dtype=tf.float32)
            sentinel_Q = tf.reshape(sentinel_Q, (1, 1, -1))
            sentinel_Q = tf.tile(sentinel_Q, (tf.shape(states_Q)[0], 1, 1))

            # Apply a linear layer with tanh nonlinearity to the question hidden states
            states_Q = tf.tanh(tf.reshape(tf.matmul(tf.reshape(states_Q, [-1, d]), W_c), [-1, M, d]) + b_c)

            # Add sentinel vectors to both context and question states, left concatenate  
            # Make it possible to atten to none of the provided hidden states
            states_P = tf.concat([sentinel_P, states_P], 1) # shape (B, N+1, d)
            states_Q = tf.concat([sentinel_Q, states_Q], 1) # shape (B, M+1, d)
            # Extend masks accordingly, 1 for sentinel vector 
            extra_bit = tf.tile(tf.constant([[1]]), (tf.shape(states_P)[0], 1))
            P_mask = tf.concat([extra_bit, P_mask], 1) # shape (B, N+1)
            Q_mask = tf.concat([extra_bit, Q_mask], 1) # shape (B, M+1)

            # Calculate attention distribution

            states_Q_t = tf.transpose(states_Q, perm=[0, 2, 1]) # (B, d, M+1)
            unmasked_affinity = tf.matmul(states_P, states_Q_t) # shape (B, N+1, M+1)
            Q_affinity_mask = tf.expand_dims(Q_mask, 1) # shape (B, 1, M+1)
            _, C2Q_dist = masked_softmax(unmasked_affinity, Q_affinity_mask, 2) # shape (B, N+1, M+1). take softmax over questions

            # Use attention distribution to take weighted sum of values
            a = tf.matmul(C2Q_dist, states_Q) # shape (B, N+1, d)
            # Apply dropout
            a = tf.nn.dropout(a, self.keep_prob) # shape (B, N+1, d)

            P_affinity_mask = tf.expand_dims(P_mask, 2) # shape (B, N+1, 1)
            _, Q2C_dist = masked_softmax(unmasked_affinity, P_affinity_mask, 1) # shape (B, N+1, M+1). take softmax over contexts
            Q2C_dist = tf.transpose(Q2C_dist, perm=[0, 2, 1]) # shape (B, M+1, N+1)

            # Use attention distribution to take weighted sum of values
            b = tf.matmul(Q2C_dist, states_P) # shape (B, M+1, d)
            # Apply dropout
            b = tf.nn.dropout(b, self.keep_prob) # shape (B, M+1, d)

            s = tf.matmul(C2Q_dist, b) # shape (B, N+1, d) 

            inputs = tf.concat([s, a], 2) # shape (B, N+1, 2*d)
            input_lens = tf.reduce_sum(Q_mask, reduction_indices=1) # shape(B)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell_fw, self.lstm_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2) # shape (B, N+1, 4*d)

            # Remove sentinel
            out = out[:,1:,:] # shape (B, N, 4*d)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SelfAttn(object):
    """
    Module for self attention.
    """

    def __init__(self, keep_prob, hidden_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, states_P, P_mask):
        """
        
        Inputs:
          states_P: Tensor shape (batch_size, passage_size, vec_size) # [B, N, d]
          P_mask: Tensor shape (batch_size, question_size).
            1s where there's real input, 0s where there's padding

        Outputs:
          states_bm: Tensor shape (batch_size, passage_dize, 4*d).
        """
        with vs.variable_scope("SelfAttn"):

            # states_P: Tensor shape (batch_size, passage_size, 2*vec_size)
            B = states_P.shape.as_list()[0] # batch size
            N = states_P.shape.as_list()[1] # passage size, number of timesteps
            d = states_P.shape.as_list()[2] # vec size

            # Define attention parameters
            W_v_P = tf.get_variable('W_v_P', shape=[d, d], dtype=tf.float32)
            W_v_Phat = tf.get_variable('W_v_Phat', shape=[d, d], dtype=tf.float32)
            V = tf.get_variable('V', shape=[d, 1], dtype=tf.float32)

            attn_params = {'W_v_P': W_v_P, 'W_v_Phat': W_v_Phat, 'V': V}

            W_g = tf.get_variable('W_g', shape=[2*d, 2*d], dtype=tf.float32) # gate 

            # batch major -> timestep major
            states_P_t = tf.transpose(states_P, [1,0,2]) # (N, B, d)

            #initial_state = tf.zeros(dtype=tf.float32, shape=[B, 2*d])
            #states = [initial_state]
            inputs = []

            for t in range(N): # N timesteps (passage size)

                ct = self.attention_pooling(states_P, states_P_t[t], attn_params, P_mask, d, N) # [B, d]

                inp = tf.concat([states_P_t[t], ct], axis = -1) # [B, 2*d]

                # Define gate
                gate = tf.nn.sigmoid(tf.matmul(inp, W_g)) # [B, 2*d]

                # Apply gate 
                inp = gate * inp # [B, 2*d]

                inputs.append(inp)

            # timestep major -> batch major
            inputs_bm = tf.transpose(tf.stack(inputs), [1, 0, 2])    
            
            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            input_lens = tf.reduce_sum(P_mask, reduction_indices=1)
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs_bm, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            states_bm = tf.concat([fw_out, bw_out], 2) # [B, N, 4*d]

            return states_bm


    def attention_pooling(self, states_P, states_P_t, params, mask, d, N):
        # states_P: Tensor shape (B, N, d)
        # states_P_t: Tensor shape (B, d)
        # mask: Tensor shape (B, N)
        # W_u_P [d, d], # W_u_Q [d, d], # W_v_P [2*d, d]
        W_v_P, W_v_Phat = params['W_v_P'], params['W_v_Phat']
        z = tf.reshape(tf.matmul(tf.reshape(states_P, [-1, d]), W_v_P), [-1, N, d]) # [B, N, d]
        z += tf.expand_dims(tf.matmul(states_P_t, W_v_Phat), axis=1) # [B, N, d]
        a = tf.tanh(z) # [B, N, d]
        V = params['V'] # [d, 1]
        s = tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), V), [-1, N]) # [B, N]
        
        # Masking 
        exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
        masked_s = tf.add(s, exp_mask) # where there's padding, set logits to -large
        # The distribution should sum to 1, and should be 0 in the value locations that correspond to padding.
        dist = tf.nn.softmax(masked_s) # [B, M] 
        
        # Use the attention distribution as weights
        ct = tf.reduce_sum(states_P * tf.expand_dims(dist, axis=-1), axis=1) # [B, d]
        
        return ct


class PointerNetwork(object):
    """
    Module for pointer network.
    """

    def __init__(self):
        pass

    def build_graph(self, states_Q, states_P, Q_mask, P_mask):
        """
        
        
        Inputs:
          states_Q: Tensor shape (batch_size, question_size, vec_size)
          states_P: Tensor shape (batch_size, passage_size, vec_size)
          Q_mask: Tensor shape (batch_size, question_size).
          P_mask: Tensor shape (batch_size, passage_size).
            1s where there's real input, 0s where there's padding

        Outputs:
          softmax logits for the start and the end of the answer span
        """
        with vs.variable_scope("PointerNetwork"):

            # states_Q: Tensor shape (batch_size, question_size, vec_size)
            # states_P: Tensor shape (batch_size, passage_size, vec_size)
            B = states_Q.shape.as_list()[0] # batch size
            M = states_Q.shape.as_list()[1] # question size
            N = states_P.shape.as_list()[1] # passage size
            d = states_P.shape.as_list()[2] # passage vec size
            dQ = states_Q.shape.as_list()[2] # question vec size

            # Define attention parameters
            W_h_P = tf.get_variable('W_h_P', shape=[d, dQ], dtype=tf.float32)
            W_h_a = tf.get_variable('W_h_a', shape=[dQ, dQ], dtype=tf.float32)
            V = tf.get_variable('V', shape=[dQ, 1], dtype=tf.float32)

            attn_params = {'W_h_P': W_h_P, 'W_h_a': W_h_a, 'V': V}

            # Define attention cell 
            cell = rnn.GRUCell(num_units = dQ)

            # Utilize the question vector rQ as the initial state of the answer recurrent network
            initial_state = self.question_pooling(states_Q, Q_mask, dQ, M) # [B, dQ]

            # p1_logits: [B, N], p1_dist: [B, N], ct: [B, d]
            p1_logits, p1_dist, ct = self.attention_pooling(states_P, initial_state, attn_params, P_mask, d, dQ, N) 

            # state: [B, dQ]
            _, state = cell(ct, initial_state)

            tf.get_variable_scope().reuse_variables()

            # p2_logits: [B, N], p2_dist: [B, N], _: [B, d]
            p2_logits, p2_dist, _ = self.attention_pooling(states_P, state, attn_params, P_mask, d, dQ, N)

            #logits = tf.stack((p1_logits,p2_logits), 1) # [B, 2, N]
            #dist = tf.stack((p1_dist,p2_dist), 1) # [B, 2, N]

            #return logits, dist
            return p1_logits, p1_dist, p2_logits, p2_dist

    def question_pooling(self, states_Q, mask, dQ, M):
        """
        Return: 
          rQ = att(uQ, VrQ), an attention-pooling vector of the question based on the parameter VrQ
        """
        with tf.variable_scope("question_pooling"):
            V_r_Q = tf.get_variable('V_r_Q', shape=[M, dQ], dtype=tf.float32)

            W_u_Q = tf.get_variable('W_u_Q', shape=[dQ, dQ], dtype=tf.float32)
            W_v_Q = tf.get_variable('W_v_Q', shape=[dQ, dQ], dtype=tf.float32)

            z = tf.reshape(tf.matmul(tf.reshape(states_Q, [-1, dQ]), W_u_Q), [-1, M, dQ]) # [B, M, dQ]
            z += tf.expand_dims(tf.matmul(V_r_Q, W_v_Q), axis=0) # [B, M, dQ]
            a = tf.tanh(z) # [B, M, dQ]
            V = tf.get_variable('V', shape=[dQ, 1], dtype=tf.float32) # [dQ, 1]
            s = tf.reshape(tf.matmul(tf.reshape(a, [-1, dQ]), V), [-1, M]) # [B, M]
        
            # Masking 
            exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
            masked_s = tf.add(s, exp_mask) # where there's padding, set logits to -large
            # The distribution should sum to 1, and should be 0 in the value locations that correspond to padding.
            dist = tf.nn.softmax(masked_s) # [B, M] 

            # Use the attention distribution as weights
            rQ = tf.reduce_sum(states_Q * tf.expand_dims(dist, axis=-1), axis=1) # [B, dQ]
        
            return rQ

    def attention_pooling(self, states_P, prev_v, params, mask, d, dQ, N):
        # states_P: Tensor shape (B, N, d)
        # prev_v: Tensor shape (B, dQ)
        # mask: Tensor shape (B, N)
        # W_h_P [d, dQ], W_h_a [dQ, dQ]
        W_h_P, W_h_a = params['W_h_P'], params['W_h_a']
        z = tf.reshape(tf.matmul(tf.reshape(states_P, [-1, d]), W_h_P), [-1, N, dQ]) # [B, N, dQ]
        z += tf.expand_dims(tf.matmul(prev_v, W_h_a), axis=1) # [B, N, dQ]
        a = tf.tanh(z) # [B, N, dQ]
        V = params['V'] # [dQ, 1]
        s = tf.reshape(tf.matmul(tf.reshape(a, [-1, dQ]), V), [-1, N]) # [B, N]
        
        # Masking 
        exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
        masked_s = tf.add(s, exp_mask) # where there's padding, set logits to -large
        # The distribution should sum to 1, and should be 0 in the value locations that correspond to padding.
        dist = tf.nn.softmax(masked_s) # [B, N] 

        # Use the attention distribution as weights
        ct = tf.reduce_sum(states_P * tf.expand_dims(dist, axis=-1), axis=1) # [B, d]
        
        return masked_s, dist, ct
    
    
class DynamicPointingDecoder(object):
    """Module ffor Dynamic Pointing Decoder.

    """

    def __init__(self, keep_prob, state_size=100, max_iter=4, pool_size=4):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.keep_prob = keep_prob
        self.state_size = state_size
        self.max_iter = max_iter
        self.pool_size = pool_size
        self.lstm_cell = rnn_cell.LSTMCell(self.state_size)
        self.lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell, input_keep_prob=keep_prob)
        
    def build_graph(self, states_P, P_mask):
        """
    

        Inputs:
          states_P: Tensor shape (batch_size, passage_size, vec_size)
          P_mask: Tensor shape (batch_size, passage_size).
            1s where there's real input, 0s where there's padding

        Outputs:
          out: Tensor shape (, , ).
        """
        with vs.variable_scope("DynamicPointingDecoder"):
            
            B = tf.shape(states_P)[0] # batch size

            start = tf.zeros([B, ], dtype=tf.int32) # shape (B, )
            input_lens = tf.reduce_sum(P_mask, reduction_indices=1) # shape (B, )
            end = input_lens - 1 # shape (B, )
            answer = tf.stack([start, end], 1) # shape (B, 2)
            
            # initialize 
            state = self.lstm_cell.zero_state(B, dtype=tf.float32) # shape (B, state_size)

            for i in range(self.max_iter):

                span_encoding = self.get_span_encoding(answer, states_P) # shape (B, 2*d)
                output, state = self.lstm_cell(span_encoding, state) # output: shape (B, state_size)

                with tf.variable_scope('start', reuse=tf.AUTO_REUSE):
                
                    alpha, start_probdist = self.highway_maxout_network(states_P, output, answer, P_mask) # shape (B, N)

                # update start
                start = tf.argmax(alpha, axis=1, output_type=tf.int32) # shape (B, )
                # update answer
                answer = tf.stack([start, answer[:, 1]], axis=1) # shape (B, 2)

                # conditioned on start 
                with tf.variable_scope('end', reuse=tf.AUTO_REUSE):

                    beta, end_probdist = self.highway_maxout_network(states_P, output, answer, P_mask) # shape (B, N)

                logits = tf.stack([alpha, beta], axis=2) # shape (B, N, 2)
                start_logits, end_logits = logits[:, :, 0], logits[:, :, 1] # shape (B, N)
                start = tf.argmax(start_logits, axis=1, output_type=tf.int32) # shape (B, )
                end = tf.argmax(end_logits, axis=1, output_type=tf.int32) # shape (B, )
                answer = tf.stack([start, end], axis=1) # shape (B, 2)

            return start_logits, start_probdist, end_logits, end_probdist

    def get_span_encoding(self, answer, states_P):
        """
    

        Inputs:
            
         
        Outputs:
          
        """
        B = tf.shape(states_P)[0] # batch_size
        start, end = answer[:, 0], answer[:, 1] # shape (B, )
        start_encoding = tf.gather_nd(states_P, tf.stack([tf.range(B), start], axis=1)) # shape (B, d)
        end_encoding = tf.gather_nd(states_P, tf.stack([tf.range(B), end], axis=1)) # shape (B, d)
        span_encoding = tf.concat([start_encoding, end_encoding], axis=1)  # shape (B, 2*d)
        return span_encoding

    def highway_maxout_network(self, states_P, state, answer, P_mask):
        """
    

        Inputs:
            
         
        Outputs:
          
        """
        span_encoding = self.get_span_encoding(answer, states_P)

        r_input = tf.concat([state, span_encoding], axis=1) # shape (B, state_size + 2*d)
        r_input = tf.nn.dropout(r_input, self.keep_prob) # shape (B, state_size + 2*d)
        r = tf.layers.dense(r_input, self.state_size, use_bias=False, activation=tf.tanh) # shape (B, state_size)
        r = tf.expand_dims(r, 1) # shape (B, 1, state_size)
        N = states_P.shape.as_list()[1] # passage size
        r = tf.tile(r, (1, N, 1)) # shape (B, N, state_size)
                
        highway_input = tf.concat([states_P, r], axis=2) # shape (B, N, state_size + d)
        layer1 = self.maxout_layer(highway_input, self.state_size) # shape (B, N, state_size)
        layer2 = self.maxout_layer(layer1, self.state_size) # shape (B, N, state_size)
        highway = tf.concat([layer1, layer2], -1) # shape (B, N, state_size*2)
        output = self.maxout_layer(highway, 1) # shape (B, N, 1)
        output = tf.squeeze(output, -1) # shape (B, N)
        masked_logits, prob_dist = masked_softmax(output, P_mask, 1) # shape (B, N)

        return masked_logits, prob_dist


    def maxout_layer(self, inputs, hidden_size):
        """
    

        Inputs:
          inputs: Tensor shape (, , )
          hidden_size: Hidden units of highway maxout network.  
         
        Outputs:
          output: Tensor shape (, , )
        """
        inputs = tf.nn.dropout(inputs, self.keep_prob)
        pool = tf.layers.dense(inputs, hidden_size*self.pool_size) # shape (B, N, state_size * pool_size)
        pool = tf.reshape(pool, (-1, inputs.get_shape()[1], hidden_size, self.pool_size)) # shape (B, N, state_size, pool_size)
        output = tf.reduce_max(pool, -1) # shape (B, N, state_size)
        return output
    
    
class BiMPM(object):
    """Module for Bilateral Multi-Perspective Matching.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

    def build_graph(self, context_fw, context_bw, question_fw, question_bw, context_mask, question_mask):
    #def build_graph(self, context_reps, question_reps, context_mask, question_mask):
        """

        Inputs:
          context_fw: Tensor shape (batch_size, context_len, hidden_size).
          context_bw: Tensor shape (batch_size, context_len, hidden_size).
          question_fw: Tensor shape (batch_size, question_len, hidden_size).
          question_bw: Tensor shape (batch_size, question_len, hidden_size).
          context_mask: Tensor shape (batch_size, context_len).
          question_mask: Tensor shape (batch_size, question_len).

        Outputs:
          question_aware_dim: 8*MP_dim
          question_aware_reps: Tensor shape (batch_size, context_len, 8*MP_dim).
            ...
        """
        with vs.variable_scope("BiMPM"):
              
            # C2Q
            question_aware_reps = []
            question_aware_dim = 0
              
            # match_reps_fw: shape(B, N, 4*MP_dim)
            # match_dim_fw: 4*MP_dim
            (match_reps_fw, match_dim_fw) = self.match_passage_with_question(context_fw, question_fw, context_mask, question_mask, forward=True)
            question_aware_reps.append(match_reps_fw)
            question_aware_dim += match_dim_fw
            # match_reps_bw: shape(B, N, 4*MP_dim)
            # match_dim_bw: 4*MP_dim
            (match_reps_bw, match_dim_bw) = self.match_passage_with_question(context_bw, question_bw, context_mask, question_mask, forward=False)
            question_aware_reps.append(match_reps_bw)
            question_aware_dim += match_dim_bw
            
            #(match_reps, match_dim) = self.match_passage_with_question(context_reps, question_reps, context_mask, question_mask, forward=True)
            #question_aware_reps.append(match_reps)
            #question_aware_dim += match_dim
            
            question_aware_reps = tf.concat(axis=2, values=question_aware_reps) # shape (batch_size, context_len, l)
            
            # Q2C
            context_aware_reps = []

            #(match_reps_Q2C, match_dim_Q2C) = self.match_passage_with_question(question_reps, context_reps, question_mask, context_mask, forward=True)
            #context_aware_reps.append(match_reps_Q2C)
            #context_aware_reps = tf.concat(axis=2, values=context_aware_reps)  #(batch_size, question_len, l)
            (match_reps_fw, match_dim_fw) = self.match_passage_with_question(question_fw, context_fw, question_mask, context_mask, forward=True)
            context_aware_reps.append(match_reps_fw)
            (match_reps_bw, match_dim_bw) = self.match_passage_with_question(question_bw, context_bw, question_mask, context_mask, forward=False)
            context_aware_reps.append(match_reps_bw)
            context_aware_reps = tf.concat(axis=2, values=context_aware_reps)

            context_aware_reps = tf.reduce_max(context_aware_reps, axis=1) # shape (batch_size, l)
            #context_aware_reps = tf.reduce_mean(context_aware_reps, axis=1) # shape (batch_size, l)
            context_aware_reps = tf.expand_dims(context_aware_reps, axis=1) # shape (batch_size, 1, l)
            final_reps = tf.multiply(question_aware_reps, context_aware_reps) # # shape (batch_size, context_len, l)

            return final_reps, question_aware_dim
            #return question_aware_reps, question_aware_dim

    def match_passage_with_question(self, context_reps, question_reps, context_mask, question_mask, MP_dim = 10,
        with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True, forward=True):
        context_reps = tf.multiply(context_reps, tf.expand_dims(tf.cast(context_mask, tf.float32),-1))
        question_reps = tf.multiply(question_reps, tf.expand_dims(tf.cast(question_mask, tf.float32),-1))

        all_question_aware_reps = []
        dim = 0

        with tf.variable_scope("match_passage_with_question"):

            if with_full_match:
                if forward:
                    question_full_rep = question_reps[:,-1,:] # first step: shape(B, d)
                else: # backward
                    question_full_rep = question_reps[:,0,:] # last step shape(B, d)

                N = tf.shape(context_reps)[1]
                question_full_rep = tf.expand_dims(question_full_rep, axis=1) # shape(B, 1, d)
                question_full_rep = tf.tile(question_full_rep, [1, N, 1]) # shape(B, N, d)

                (reps, match_dim) = self.multi_perspective_matching(context_reps, question_full_rep, scope='mp-match-full-match')
                all_question_aware_reps.append(reps)
                dim += match_dim

            if with_maxpool_match:
                maxpooling_rep = self.maxpooling_matching(context_reps, question_reps)
                all_question_aware_reps.append(maxpooling_rep)
                #dim += MP_dim * 2
                dim += MP_dim * 2

            if with_attentive_match:
                questions_t = tf.transpose(question_reps, perm=[0, 2, 1]) # (B, d, M)
                attn_logits = tf.matmul(context_reps, questions_t) # shape (B, N, M)
                attn_logits_mask = tf.expand_dims(question_mask, 1) # shape (B, 1, M)
                _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (B, N, M). take softmax over values

                # Use attention distribution to take weighted sum of values
                attn_reps = tf.matmul(attn_dist, question_reps) # shape (B, N, d)
                (reps, match_dim) = self.multi_perspective_matching(context_reps, attn_reps, scope='mp-match-attentive-match')
                all_question_aware_reps.append(reps)
                dim += match_dim

            if with_max_attentive_match:
                questions_t = tf.transpose(question_reps, perm=[0, 2, 1]) # (B, d, M)
                attn_logits = tf.matmul(context_reps, questions_t) # shape (B, N, M)
                attn_logits_mask = tf.expand_dims(question_mask, 1) # shape (B, 1, M)
                _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (B, N, M). take softmax over values

                # Use attention distribution to collect max
                positions = tf.argmax(attn_dist, axis=2, output_type=tf.int32)  # shape (B, N)

                B = tf.shape(question_reps)[0]
                N = tf.shape(positions)[1]
                batch_nums = tf.range(0, limit=B) # shape (B,)
                batch_nums = tf.reshape(batch_nums, shape=[-1, 1]) # shape (B, 1)
                batch_nums = tf.tile(batch_nums, multiples=[1, N]) # shape (B, N)

                indices = tf.stack((batch_nums, positions), axis=2) # shape (B, N, 2)
                max_attn_reps = tf.gather_nd(question_reps, indices)
                
                (reps, match_dim) = self.multi_perspective_matching(context_reps, max_attn_reps, scope='mp-match-max-attentive-match')
                all_question_aware_reps.append(reps)
                dim += match_dim

            all_question_aware_reps = tf.concat(axis=2, values=all_question_aware_reps)

        return (all_question_aware_reps, dim)

    def multi_perspective_matching(self, reps1, reps2, scope, MP_dim = 10):

        # l: cosine_MP_dim
        matching_result = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            mp_cosine_params = tf.get_variable('mp_W', shape=[MP_dim, self.hidden_size], dtype=tf.float32) # shape(l, d)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0) # shape(1, l, d)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0) # shape(1, 1, l, d)
            
            reps1_flat = tf.expand_dims(reps1, axis=2) # shape(B, N, 1, d)
            reps2_flat = tf.expand_dims(reps2, axis=2) # shape(B, N, 1, d)
            
            mp_cosine_matching = self.cosine_distance(tf.multiply(reps1_flat, mp_cosine_params),reps2_flat) # shape(B, N, l)

            matching_result.append(mp_cosine_matching)

            matching_result = tf.concat(axis=2, values=matching_result)

        return (matching_result, MP_dim)

    def maxpooling_matching(self, context_reps, question_reps, MP_dim = 20):
        def single_instance(elem):
            p = elem[0]
            q = elem[1]
            # p: [context_len, d], q: [question_len, d], params: [cosine_MP_dim, d]
            with tf.variable_scope('maxpooling_matching', reuse=tf.AUTO_REUSE):
                maxpooling_params = tf.get_variable('maxpooling_W',shape=[MP_dim, self.hidden_size], dtype=tf.float32)
                params = tf.expand_dims(maxpooling_params, axis=0) # [1, cosine_MP_dim, d]
                p = tf.expand_dims(p, axis=1) #[context_len, 1, d]
                q = tf.expand_dims(q, axis=1) #[question_len, 1, d]
            
                # element-wise multiplication
                p = tf.multiply(p, params) # [context_len, cosine_MP_dim, d]
                q = tf.multiply(q, params) # [question_len, cosine_MP_dim, d]

                p = tf.expand_dims(p, 1) # [context_len, 1, cosine_MP_dim, d]
                q = tf.expand_dims(q, 0) # [1, question_len, cosine_MP_dim, d]
                return self.cosine_distance(p, q) # [passage_len, question_len, cosine_MP_die]

        elems = (context_reps, question_reps)
        matching_result = tf.map_fn(single_instance, elems, dtype=tf.float32) # [batch_size, passage_len, question_len, cosine_MP_dim]
        return tf.concat(axis=2, values=tf.reduce_max(matching_result, axis=2))# [batch_size, passage_len, cosine_MP_dim]

    def cosine_distance(self, v1, v2):
        eps = 1e-6
        numerator = tf.reduce_sum(tf.multiply(v1, v2), axis=-1)
        v1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1), axis=-1), eps))
        v2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2), axis=-1), eps)) 
        return numerator / v1_norm / v2_norm
