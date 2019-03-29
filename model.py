# codeing: utf-8

import mxnet as mx
import numpy as np
from mxnet.gluon import HybridBlock
from mxnet import gluon


class CNNTextClassifier(HybridBlock):

    def __init__(self, emb_input_dim, emb_output_dim, num_classes=2, prefix=None, params=None):
        super(CNNTextClassifier, self).__init__(prefix=prefix, params=params)
        
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)  # embedding layer
            # self.encoder = .. a series of convolutional layers with pooling
            self.encoder = gluon.nn.HybridSequential()
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Conv2D(50, kernel_size=(3, emb_output_dim), activation='relu'))  # conv layer with 50 kernels
                self.encoder.add(gluon.nn.GlobalMaxPool2D())  # global max pooling layer
            
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(0.5))  # dropout layer
                self.output.add(gluon.nn.Dense(num_classes))    # fully connected dense layer

    def hybrid_forward(self, F, data):
        embedded = self.embedding(data)
        encoded = self.encoder(F.reshape(embedded, (-1, 1, 64, 100)))
        return self.output(encoded)
