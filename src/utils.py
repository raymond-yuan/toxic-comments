from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors, FastText
import os, codecs
from config import *

def load_fasttext_embeddings_lim(embeddings_path, word_index, max_features=100000):
    embeddings_index = {}
    missing = set()
    with codecs.open(embeddings_path, encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and len(embedding_vector) > 0:
             embedding_matrix[i] = embedding_vector
        else:
            missing.add(word)
    print('total number of words in word index: {}'.format(len(word_index)))
    print('number of null word embeddings: {}'.format(np.sum(
                                    np.sum(embedding_matrix, axis=1) == 0)))
    print('Number of missing words: {}'.format(len(missing)))
    return embedding_matrix

def load_w2v_embeddings(embeddings_path, word_index, max_features=100000):
    extension = os.path.splitext(embeddings_path)[1]
    is_binary = extension == ".bin"
    print("Reading in", "binary" if is_binary else "text", "Word2Vec embeddings")
    word_vectors = KeyedVectors.load_word2vec_format(embeddings_path, binary=is_binary)
    embedding_dim = word_vectors.vector_size

    # Now create the embeddings matrix
    nb_words = min(max_features, len(word_index))
    embeddings = np.zeros((nb_words, embedding_dim))
    missing = set()
    for word, i in tqdm(word_index.items()):
        if i >= max_features: continue
        if word in word_vectors.vocab:
            embeddings[i] = word_vectors[word]
        else:
            missing.add(i)
    print("Loaded", len(embeddings), "Word2vec embeddings with", len(missing), "missing")
    return embeddings, missing

def load_fasttext_embeddings(embeddings_path, word_index):
    print("Reading in FastText embeddings from " + embeddings_path)
    try:
        word_vectors = FastText.load_fasttext_format(embeddings_path)
    except NotImplementedError:
        print('Something went wrong in loading fasttext format')
        word_vectors = FastText.load(embeddings_path)
    embedding_dim = word_vectors.vector_size
    # Now create the embeddings matrix
    embeddings = np.zeros((len(word_index) + 1, embedding_dim))
    missing = set()
    for word, i in tqdm(word_index.items()):
        if word in word_vectors:
            embeddings[i] = word_vectors[word]
        else:
            missing.add(i)
    print("Loaded", len(embeddings), "FastText embeddings with", len(missing), "missing")
    return embeddings, missing

def save_embeddings(embeddings, missing, save_dir="../embeddings/"):
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
    print("Saved the embeddings")
    with open(os.path.join(save_dir, "missing.pkl"), 'wb') as missing_file:
        pkl.dump(missing, missing_file)
    print("Saved missing indicies")

def ensemble_submissions(in_list):
    return sum(np.array(in_list)) / float(len(in_list))

if __name__ == '__main__':
    with open(word_idex_path, 'rb') as word_index_file:
        word_index = pkl.load(word_index_file)

    embeddings_mat, missing = load_fasttext_embeddings(EMBEDDING_FILE, word_index)
    save_embeddings(embeddings_mat, missing)
