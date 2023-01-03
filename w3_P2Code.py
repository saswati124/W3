## Saswati Mahapatra
## 2022 Copyright Reserved

from __future__ import print_function
import numpy as np
import pandas as pd
#import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import metrics
from random import seed
import time
import sys
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior() 

tf.reset_default_graph()

ppi_adj = np.loadtxt("Example_PPIEdgelist_adjacency.csv", dtype=int, delimiter=",") #adjacency matrix of PPI network of target genes
TOM_adj = np.loadtxt("Example_GCN_adjacency.csv", dtype=int, delimiter=",") #adjacency matrix of TOM GCN network of target genes
expression = np.loadtxt("Example_targetGeneExpr.csv", dtype=float, delimiter=",")#expression is expression matrix of target genes

partition=np.logical_and(ppi_adj,TOM_adj)

label_vec = np.array(expression[:,-1], dtype=int)  #separating the class label column only
expression = np.array(expression[:,:-1])  #Expression level for 500 genes 

labels = []
for l in label_vec:
    if l == 1:
        labels.append([0,1])
    else:
        labels.append([1,0])
labels = np.array(labels,dtype=int)

## train/test data split
cut = int(0.75*np.shape(expression)[0])
expression, labels = shuffle(expression, labels)
x_train = expression[:cut, :]
x_test = expression[cut:, :]
y_train = labels[:cut, :]
y_test = labels[cut:, :]

## hyper-parameters and settings
L2 = False
learning_rate = 0.0001
training_epochs = 50
batch_size = 8
display_step = 1

n_hidden_1 = np.shape(partition)[0]  # 0 for row index. Here partition is a square adjacency matrix. 
n_hidden_2 = 64
n_classes = 2
n_features = np.shape(expression)[1] #i.e the number of genes. 1 is for column index

## initiate training logs
loss_rec = np.zeros([training_epochs, 1]) #created a zero matrix of size training_epochs X 1 
training_eval = np.zeros([training_epochs, 2]) #created a zero matrix of size training_epochs X 2 

def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, tf.multiply(weights['h1'], partition)), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob=keep_prob)

    
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
    
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.int32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

weights = {
    'h1': tf.Variable(tf.truncated_normal(shape=[n_features, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal(shape=[n_hidden_2, n_classes], stddev=0.1))

}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

## Evaluation
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
y_score = tf.nn.softmax(logits=pred)

var_left = tf.reduce_sum(tf.abs(tf.multiply(weights['h1'], partition)), 0)
var_right = tf.reduce_sum(tf.abs(weights['h2']), 1)
var_importance = tf.add(var_left, var_right)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    total_batch = int(np.shape(x_train)[0] / batch_size)

    ## Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        x_tmp, y_tmp = shuffle(x_train, y_train)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x, batch_y = x_tmp[i*batch_size:i*batch_size+batch_size], \
                                y_tmp[i*batch_size:i*batch_size+batch_size]

            _, c= sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y,
                                                        keep_prob: 0.2,
                                                        lr: learning_rate
                                                        })
            # Compute average loss
            avg_cost += c / total_batch

        del x_tmp
        del y_tmp

        ## Display logs per epoch step
        if epoch % display_step == 0:
            loss_rec[epoch] = avg_cost
            acc, y_s = sess.run([accuracy, y_score], feed_dict={x: x_train, y: y_train, keep_prob: 1})
            auc = metrics.roc_auc_score(y_train, y_s)
            training_eval[epoch] = [acc, auc]
            print ("Epoch:", '%d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost),
                    "Training accuracy:", round(acc,3), " Training auc:", round(auc,3))

        if avg_cost <= 0.1:
            print("Early stopping.")
            break

    ## Testing cycle
    acc, y_s = sess.run([accuracy, y_score], feed_dict={x: x_test, y: y_test, keep_prob:1})
    test_auc = metrics.roc_auc_score(y_test, y_s)
    var_imp = sess.run([var_importance])
    var_imp = np.reshape(var_imp, [n_features])
    print("*****=====", "Testing accuracy: ", acc, " Testing auc: ", auc, "=====*****")
    np.savetxt("Example_Feature_imp_GFDNN.csv", var_imp, delimiter=",") #saving the feature importance in a csv file
    
    
