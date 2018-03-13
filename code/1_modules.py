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

class charCNN(object):
    """
    A CNN for character-level emmbeddings
    """
    def __init__(self, keep_prob, ):
        pass

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
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

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
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


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
            W_v_P = tf.get_variable('W_v_Q', shape=[2*d, d], dtype=tf.float32)
            V = tf.get_variable('V', shape=[d, 1], dtype=tf.float32)

            attn_params = {'W_u_Q': W_u_Q, 'W_u_P': W_u_P, 'W_v_P': W_v_P, 'V': V}

            W_g = tf.get_variable('W_g', shape=[2*d, 2*d], dtype=tf.float32) # gate 

            # Define attention cell 
            cell = rnn.GRUCell(num_units = 2*d)

            # batch major -> timestep major
            states_P = tf.transpose(states_P, [1,0,2]) # (N, B, d)

            initial_state = tf.zeros(dtype=tf.float32, shape=[100, 2*d])
            states = [initial_state]

            for t in range(N): # N timesteps (passage size)
                if t > 0:
                    tf.get_variable_scope().reuse_variables()

                ct = self.attention_pooling(states_Q, states_P[t], states[-1], attn_params, Q_mask, d, M) # [B, d]

                inputs = tf.concat([states_P[t], ct], axis = -1) # [B, 2*d]

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