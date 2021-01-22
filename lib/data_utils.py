import os

# download dataset from kaggle
def download_dataset():
    os.system('kaggle competitions download -c jigsaw-toxic-comment-classification-challenge')
    return