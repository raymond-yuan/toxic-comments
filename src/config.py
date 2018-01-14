from datetime import datetime

TRAIN_DATA_FILE = "../data/train.csv"
TEST_DATA_FILE = "../data/test.csv"
# EMBEDDING_FILE = "../data/glove.6B.50d.txt"
EMBEDDING_FILE = "../data/wiki.simple.vec"

model_name = 'GRU_Ensemble'
MODEL_DIR = "./models/{}-{}/".format(model_name, str(datetime.now()))

max_features = 20000  # Number of unique words
maxlen = 100  # max number of words in a comment to use
embed_size = 300  # Size of each word vector (default value)

batch_size = 32
epochs = 2

ensemble_num = 3
