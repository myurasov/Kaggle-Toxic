import unittest

import numpy as np
from src.config import config


class Validate_Bert_DATA(unittest.TestCase):
    def test_bert_data(self):

        data = np.fromfile(config["DATA_DIR"] + "/processsed_for_bert/test.npy")
        print(data.keys)

        pass


if __name__ == "__main__":
    unittest.main(warnings="ignore")
