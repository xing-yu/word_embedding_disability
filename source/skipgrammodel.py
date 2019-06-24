import tensorflow as tf
import numpy as np


class SkipGramModel:
    
    # build the graph for word2vec model
    def __init__(self, 
                 gen, 
                 vocab_size, 
                 batch_size, 
                 embed_size, 
                 num_sampled, 
                 learning_rate = 0.01, 
                 transfer = False, 
                 pretrain = None):
        
        
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.transfer = transfer
        self.pretrain = pretrain

    
    def _import_data(self):
        self.dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), (tf.TensorShape([self.batch_size]), tf.TensorShape([self.batch_size, 1])))
        
        self.iterator = self.dataset.make_initializable_iterator()

        self.center_words, self.target_words = self.iterator.get_next()
    
    def _create_embedding(self):
        
        with tf.name_scope('embed'):
            
            if self.transfer == False:
                
                self.embed_matrix = tf.get_variable(name = "embed_matrix", 
                                                shape = [self.vocab_size, self.embed_size], 
                                                initializer = tf.random_uniform_initializer())
            
            else:
                
                self.embed_matrix = tf.get_variable(name = "embed_matrix", 
                                                shape = [self.vocab_size, self.embed_size], 
                                                initializer = tf.constant_initializer(self.pretrain))
                
                
                
    def _create_loss(self):
        # NCE loss
        with tf.name_scope('loss'):
            
            embed = tf.nn.embedding_lookup(self.embed_matrix, 
                                           self.center_words, 
                                           name = 'embedding')
            
            nce_weight = tf.get_variable(name = 'nce_weight', 
                                         shape = [self.vocab_size, self.embed_size],
                                         initializer = tf.truncated_normal_initializer(stddev = 1.0 / (self.embed_size ** 0.5)))
            
            nce_bias = tf.get_variable(name = 'nce_bias', initializer = tf.zeros([self.vocab_size]))
            
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weight, 
                                                      biases = nce_bias, 
                                                      labels = self.target_words, 
                                                      inputs = embed, 
                                                      num_sampled = self.num_sampled, 
                                                      num_classes = self.vocab_size), 
                                       name = 'loss')
    
    def _create_optimizer(self):
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
    def build_graph(self):
        self._import_data()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()