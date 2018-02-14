from datetime import datetime

TRAIN_DATA_FILE = "../data/train.csv"
TEST_DATA_FILE = "../data/test.csv"
# EMBEDDING_FILE = "../data/glove.6B.50d.txt"
# EMBEDDING_FILE = "../data/wiki.simple.vec"
EMBEDDING_FILE = "../data/wiki.en.vec"

embed_type = EMBEDDING_FILE.split('/')[-1]

model_name = 'GRU_Ensemble'
embedding_type = 'fasttextLim'
MODEL_DIR = "./submissions/{}-{}-{}-{}/".format(embedding_type, embed_type, model_name, str(datetime.now()).replace(' ', '-'))

data_augmentors = ["train_de.csv", "train_fr.csv", "train_es.csv"]
# data_augmentors = []
# MODEL_DIR = '/home/raymond/Documents/projects/toxic-comments/src/submissions/fasttextLim-wiki.en.vec-GRU_Ensemble-2018-02-09 22:19:07.545168'
# max_features = 100000  # Number of unique words
maxlen = 200  # max number of words in a comment to use
embed_size = 300  # Size of each word vector (default value)

batch_size = 128
epochs = 6

ensemble_num = 1
