import numpy as np
import tensorflow as tf

def data_check(data,max_time,input_size):

    if len(data) != input_size*max_time:
        raise ValueError('The data length does not match the size.')

    inputs = np.zeros([1,max_time,input_size])
    for j in range(max_time):
        inputs[0,j,:] = data[j*input_size:(j+1)*input_size]

    return inputs

def data_resturcture(data,max_time,input_size,output_size):

    batch_size = len(data) - input_size*max_time - output_size + 1
    if batch_size < 1:
        raise ValueError('The data length does not match the size.')

    inputs = np.zeros([batch_size,max_time,input_size])
    labels = np.zeros([batch_size,output_size])
    for i in range(batch_size):
        for j in range(max_time):
            inputs[i,j,:] = data[i+j*input_size:i+(j+1)*input_size];
        labels[i,:] = data[i+max_time*input_size:i+max_time*input_size+output_size];

    return inputs, labels

def LSTM_constructor(inputs,layers_units,state_keep_probs):
    # layers_sizes is a sequence of positive integers

    def _create_one_cell(units,state_keep_prob):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(units)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell,state_keep_prob=state_keep_prob)
        return lstm_cell

    rnn_layers = [_create_one_cell(units,state_keep_prob)
                  for units,state_keep_prob in zip(layers_units,state_keep_probs)]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    outputs, _ = tf.nn.dynamic_rnn(multi_rnn_cell,inputs,dtype=tf.float32)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.gather(outputs, outputs.shape[0].value-1 )

    return outputs

class LSTMPredict(object):

    def __init__(self, graph,
                 max_time, input_size,
                 layers_units, state_keep_probs,
                 learning_rate):

        self.graph = graph
        self.max_time = max_time
        self.input_size = input_size
        self.layers_units = layers_units
        self.state_keep_probs = state_keep_probs
        self.learning_rate = learning_rate

        with graph.as_default():
            sess = tf.Session()
            inputs = tf.placeholder(tf.float32,(None,max_time,input_size))
            outputs = LSTM_constructor(inputs,layers_units,state_keep_probs)
            labels = tf.placeholder(tf.float32,(None,outputs.shape[1].value))
            loss = tf.sqrt(tf.reduce_mean(tf.square(outputs-labels)))
            optim = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
            sess.run(tf.global_variables_initializer())
            
        self.sess =  sess
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        self.loss = loss
        self.optim = optim

    def _train(self,data):
        inputs, labels = data_resturcture(data,self.max_time,self.input_size,self.outputs.shape[1].value)
        loss, outputs, _ = self.sess.run(
            [self.loss, self.outputs, self.optim], {self.inputs:inputs,self.labels:labels})
        return loss, outputs, labels

    def _test(self,data):
        inputs, labels = data_resturcture(data,self.max_time,self.input_size,self.outputs.shape[1].value)
        loss, outputs = self.sess.run(
            [self.loss, self.outputs], {self.inputs:inputs,self.labels:labels})
        return loss, outputs, labels

    def _predict(self,data):
        inputs = data_check(data,self.max_time,self.input_size)
        outputs = self.sess.run(self.outputs, {self.inputs:inputs})
        return outputs[0]




