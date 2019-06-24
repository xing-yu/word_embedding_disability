import sys
sys.path.append('.')
from skipgrammodel import *
import json
from nltk.stem import PorterStemmer
import tensorflow as tf
import numpy as np

dictionary_file = sys.argv[1]
index_to_words_file = sys.argv[2]
output_file = sys.argv[3]
pretrained_embed_file = sys.argv[4]
batch_size = int(sys.argv[5])
window_size = int(sys.argv[6])
embed_size = int(sys.argv[7])
num_sampled = int(sys.argv[8])
path_to_checkpoints = sys.argv[9]
num_train_steps = int(sys.argv[10])

def main(dictionary_file, 
    index_to_words_file, 
    output_file, 
    pretrained_embed_file,  
    batch_size, 
    window_size, 
    embed_size, 
    num_sampled, 
    path_to_checkpoints,
    num_train_steps = 10000):

    # reset graph if necessary
    #tf.reset_default_graph()

    # load vocabulary
    vocab = json.load(open(dictionary_file, 'r'))

    vocab_size = len(vocab)

    # load pretrained matrix
    pretrained = np.loadtxt(pretrained_embed_file)

    # initialize generator for batches
    gen = batch_gen(index_to_words_file, vocab_size, batch_size, window_size)

    # initialize model
    model = SkipGramModel(gen, vocab_size = vocab_size, batch_size = batch_size, embed_size = embed_size, num_sampled = num_sampled, transfer = True, pretrain = pretrained)

    model.build_graph()

    # train model
    step_history, loss_history, output = train(model, num_train_steps, path_to_checkpoints)

    # save loss plot
    plt.title("Loss Function")
    plt.plot(step_history, loss_history)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig("loss.png", dpi = 150)

    # save embedding matrix
    np.savetxt(output_file, output)

# ---- generate batch ----
def batch_gen(filename, vocab_size, batch_size, window_size):
    
    # load index to words file
    
    index_words = []

    with open(filename, 'r') as f:

        for line in f:

            for v in line.strip().split():

                index_words.append(int(v))

    single_gen = generate_sample(index_words, window_size)
    
    while True:
        
        try:
        
            center_batch = np.zeros(batch_size, dtype = np.int32)

            target_batch = np.zeros([batch_size, 1])

            for index in range(batch_size):

                center_batch[index], target_batch[index] = next(single_gen)

            yield center_batch, target_batch
            
        except StopIteration:
            
            break

# ---- generate (center, target) samples ----
def generate_sample(index_words, context_window_size):
    
    import random
    
    for idx, center in enumerate(index_words):
        
        context = random.randint(1, context_window_size)
        
        # create a pair before the center word
        for target in index_words[max(0, idx - context) : idx]:
            
            yield center, target
            
        # create a pair after the center word
        for target in index_words[idx + 1 : idx + context + 1]:
            
            yield center, target

# ---- function to train model ----
def train(model, num_train_steps, path_to_checkpoints):
    
    saver = tf.train.Saver()

    # save for plotting
    loss_history = []
    step_history = []

    # set session configration for multi cpu and threads
    config = tf.ConfigProto(device_count = {"CPU": 1})
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 1
    
    with tf.Session(config = config) as sess:
        
        # initialize data and variables
        sess.run(model.iterator.initializer)
        sess.run(tf.global_variables_initializer())

        # load checkpoints if any
        ckpt = tf.train.get_checkpoint_state(path_to_checkpoints)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        # optimize
        total_loss = 0.0

        for index in range(num_train_steps):

            try:
                loss_batch, _ = sess.run([model.loss, model.optimizer])
                total_loss += loss_batch
                
                # save every 500 step
                if (index + 1) % 1000 == 0:

                    print("Loss at step {}: {:.3f}".format(index + 1, total_loss))
                    
                    saver.save(sess, path_to_checkpoints + '/word2vec', global_step = index + 1)
                    
                    loss_history.append(total_loss)
                    step_history.append(index)

                    total_loss = 0
            
            # reinitialize iterator 
            except tf.errors.OutOfRangeError:

                sess.run(model.iterator.initializer)
        
        # save final output
        final_embed_matrix = sess.run(model.embed_matrix)
                
    return step_history, loss_history, final_embed_matrix

# ---- call main ----
main(dictionary_file, 
    index_to_words_file, 
    output_file, 
    pretrained_embed_file,  
    batch_size, 
    window_size, 
    embed_size, 
    num_sampled, 
    path_to_checkpoints,
    num_train_steps)