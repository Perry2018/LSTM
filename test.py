import tensorflow as tf
import LSTMPredict as lstm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data (stock index)

url = 'DJI.csv'
df = pd.read_csv(url,index_col=0)

data = np.array(df['Close'])

n_data = data.size
data = data[1:n_data]/data[0:n_data-1] - 1

n_data -= 1
n_train_data = 1000
n_test_data = 200

train_data = data[n_data-n_test_data-n_train_data:n_data-n_test_data]
test_data = data[n_data-n_test_data:n_data]

# make LSTM neural networks

graph = tf.Graph()
max_time = 12
input_size = 5
layers_units = [20,1]
state_keep_probs = [0.5,0.5]
learning_rate = 0.5

model = lstm.LSTMPredict(graph,
                 max_time, input_size,
                 layers_units, state_keep_probs,
                 learning_rate)

# conduct tests

ylim = 0.1

print('train:')
n_train = 10
for i in range(n_train):
    loss, _, _ = model._train(train_data)
    print((i, loss))

loss, outputs, labels = model._test(train_data)
print((n_train,loss))

outputs = outputs.reshape(outputs.size)
labels = labels.reshape(labels.size)
print(np.std(outputs))
print(np.std(labels))

plt.figure()
plt.title('Train')
plt.plot(outputs,'r',label='predictions')
plt.plot(labels,'b',label='real data')
plt.ylim((-ylim, ylim))
plt.ylabel('stock return')
plt.legend(loc='best')
plt.show()

print('test:')
loss, outputs, labels = model._test(test_data)
print(loss)

outputs = outputs.reshape(outputs.size)
labels = labels.reshape(labels.size)
print(np.std(outputs))
print(np.std(labels))

plt.figure()
plt.title('Test')
plt.plot(outputs,'r',label='predictions')
plt.plot(labels,'b',label='real data')
plt.ylim((-ylim, ylim))
plt.ylabel('stock return')
plt.legend(loc='best')
plt.show()

