from datetime import datetime

TRAIN_DATA_FILE = "../data/train.csv"
TEST_DATA_FILE = "../data/test.csv"
# EMBEDDING_FILE = "../data/glove.6B.50d.txt"
# EMBEDDING_FILE = "../data/wiki.simple.vec"
EMBEDDING_FILE = "../data/wiki.en.bin"

embed_type = EMBEDDING_FILE.split('/')[-1]

# model_name = 'GRU_Ensemble'
model_name = 'GRU_CUDNN'
embedding_type = 'fasttextLim'

load_embed = '/home/raymond/Documents/projects/toxic-comments/data/fasttext-embeddings/wiki.en.bin.fasttext-257356.npz'

testing = False
n_splits = 10
if testing:
    print('TESTING')
    MODEL_DIR = "./submissions/{}-{}-{}-{}/".format(embedding_type, embed_type, model_name, 'TEST')
else:
    MODEL_DIR = "./submissions/{}-{}-{}-{}/".format(embedding_type, embed_type,
                                            model_name, str(datetime.now()).replace(' ', '-'))

data_augmentors = ["train_de.csv", "train_fr.csv", "train_es.csv"]
# data_augmentors = []
# MODEL_DIR = '/home/ubuntu/toxic-comments/src/submissions/fasttext-wiki.en.bin-GRU_Ensemble-2018-02-14-14:59:50.685781/'
# max_features = 100000  # Number of unique words
maxlen = 500  # max number of words in a comment to use
embed_size = 300  # Size of each word vector (default value)

batch_size = 16
epochs = 7

pad_batches = True
