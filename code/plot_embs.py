import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1 


def get_glove(glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path

    Returns:
      emb_matrix: Numpy array shape (400002, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """

    print "Loading GLoVE vectors from file: %s" % glove_path
    vocab_size = int(4e5) # this is the vocab size of the corpus we've downloaded

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word


def restore_retrained(indices):
    result = np.zeros((25, 300))
    with tf.Graph().as_default():
        saver = tf.train.import_meta_graph('/Users/yuxing/Desktop/Stanford/Academic/2017-2018/Winter2018/CS224N/Project/Analysis/retrain/ver_026_retrain/qa.ckpt-3000.meta')
        with tf.Session() as sess:
            saver.restore(sess, "/Users/yuxing/Desktop/Stanford/Academic/2017-2018/Winter2018/CS224N/Project/Analysis/retrain/ver_026_retrain/qa.ckpt-3000")
            graph = tf.get_default_graph()
            emb_matrix = graph.get_tensor_by_name("QAModel/embeddings/emb_matrix:0")
            for i, idx in enumerate(indices):
                result[i,:] = np.array(sess.run(emb_matrix[idx]))
            return result


def plotting(visualizeIdx, visualizeVecs_glove, visualizeVecs_retrained, visualizeWords, save_name):
    print "Start plotting ... "
    visualizeVecs = np.concatenate((visualizeVecs_glove, visualizeVecs_retrained))
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    l = len(visualizeWords)

    for i in xrange(l):
        plt.text(coord[i,0], coord[i,1], visualizeWords[i],
            bbox=dict(facecolor='green', alpha=0.1))
    
    for j in xrange(l):
        plt.text(coord[j+l,0], coord[j+l,1], visualizeWords[j],
            bbox=dict(facecolor='blue', alpha=0.1))

    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

    plt.savefig(save_name)
    

def main():
    # generate random indices
    ids = []
    # for x in range(25):
    #     ids.append(random.randint(0,400001))
    # Load embedding matrix and vocab mappings
    print "Loading glove ..."
    emb_matrix, word2id, id2word = get_glove('./data/glove.6B.300d.txt', 300)

    print "Look up word indices."
    words = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"]
    for x in words:
        ids.append(word2id[x])
    print ids
    
    glove_vectors = emb_matrix[ids]

    print "Loading retrained word embedings ..."
    retrained_vectors = restore_retrained(ids)

    plotting(ids, glove_vectors, retrained_vectors, words, "word_embeddings.png")
    # plotting(ids, retrained_vectors, words, "retrained.png")



if __name__ == "__main__":
    main()