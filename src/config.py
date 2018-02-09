from datetime import datetime

TRAIN_DATA_FILE = "../data/train.csv"
TEST_DATA_FILE = "../data/test.csv"
# EMBEDDING_FILE = "../data/glove.6B.50d.txt"
# EMBEDDING_FILE = "../data/wiki.simple.vec"
EMBEDDING_FILE = "../data/wiki.en.vec"

embed_type = EMBEDDING_FILE.split('/')[-1]

model_name = 'GRU_Ensemble'
MODEL_DIR = "./submissions/{}-{}-{}/".format(embed_type, model_name, str(datetime.now()))

# max_features = 100000  # Number of unique words
maxlen = 200  # max number of words in a comment to use
embed_size = 300  # Size of each word vector (default value)

batch_size = 32
epochs = 2

ensemble_num = 1
