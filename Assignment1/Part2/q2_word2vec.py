"""Load pretrained Glove model."""

import os
import warnings
import numpy as np
from collections import OrderedDict

"""
Pre-defined oov (out-of-vocabulary) word vector.
This vector is computed by averaging all word vectors in the Glove.
See get_oov_vector function below for more information.
"""
oov_vector = np.array([-0.12920076, -0.28866628, -0.01224866, -0.05676644, -0.20210965, -0.08389011, 0.33359843, 0.16045167, 0.03867431, 0.17833012, 0.04696583, -0.00285802, 0.29099807, 0.04613704, -0.20923874, -0.06613114, -0.06822549, 0.07665912, 0.3134014, 0.17848536, -0.1225775, -0.09916984, -0.07495987, 0.06413227, 0.14441176, 0.60894334, 0.17463093, 0.05335403, -0.01273871, 0.03474107, -0.8123879, -0.04688699, 0.20193407, 0.2031118, -0.03935686, 0.06967544, -0.01553638, -0.03405238, -0.06528071, 0.12250231, 0.13991883, -0.17446303, -0.08011883, 0.0849521, -0.01041659, -0.13705009, 0.20127155, 0.10069408, 0.00653003, 0.01685157], dtype=np.float32)


class PretrainedWord2Vec(object):
    """Load pretrained Glove model and implements functions for word vectors."""

    def __init__(self, glove_path):

        if not os.path.isfile(glove_path):
            raise FileNotFoundError("Word2vec model %s not found." % glove_path)
        print("Initializing word2vec from %s .." % glove_path)

        # Initialize embeddings from glove_path
        self.embeddings = self._load_word2vec(glove_path)

    def _load_word2vec(self, glove_path):
        """
        Load word2vec to memory.
        Args:
            - glove_path: Directory for Glove model. For example, glove.6B.50d.txt.
        Return:
            OrderdDict whose key is the word and value the vector, e.g.,
                embeddings["word"] = [0.1, -0.2, ....]
            Hint: The Glove file stores data line by line. The values for each line are separated by whitespace,
                where the first value is the word, and the remaining values are the vector elements.
                IMPORTANT: convert the vectors to np.float32 type. Hint: use np.asarray(list, dtype=np.float32).
        """
        embeddings = OrderedDict()
        ### YOUR CODE HERE
        f = open(glove_path, encoding='utf-8')
        for line in f:
            splitline = line.split()
            word = splitline[0]
            coefs = np.asarray(splitline[1:], dtype='float32')
            embeddings[word] = coefs
        f.close()
        ### END YOUR CODE
        return embeddings

    @property
    def vocab_size(self):
        """Vocab size of the Glove."""
        return len(self.embeddings)

    @property
    def vocab(self):
        """List of vocabularies of the Glove."""
        return list(self.embeddings.keys())

    def health_check(self):
        """
        Check whether desired Glove (correct dimension, date type) is loaded.
        """
        queen_vector = self.get_vector("queen")
        vec_dim = queen_vector.size
        assert isinstance(self.embeddings, OrderedDict)
        assert queen_vector.dtype == "float32", "Wrong vector type, expected: np.float32."
        assert self.vocab_size == 400000, "Wrong vocabulary size, expected 400000."
        assert vec_dim == 50, "Wrong vector dimension, expected 50."
        print("Word2vec model loaded successfully.")

    def get_vector(self, word):
        """
        Get Glove vector for word.
        Args:
            - word: string.
        Return:
            Glove vector of type np.float32 of shape (50,).
            Note: If word not in the vocabulary, return the oov_vector defined above.
                DO NOT use get_oov_vector function here!
        """
        ### YOUR CODE HERE 
        if word in self.embeddings: # NOT self.embeddings.keys()!
            return self.embeddings[word]
        else:
            return oov_vector    
        ### END YOUR CODE
#         return NotImplementedError()

    def word_similarity(self, word1, word2):
        """
        Compute word similarity between word1 and word2.
        Args:
            - word1: string.
            - word2: string.
        Return:
            Similarity defined as v1^T*v2 / (||v1|| * ||v2||)
            Hint: You may consider np.linalg.norm for computing norm.
        """
        ### YOUR CODE HERE
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        matrix = np.dot(v1.T, v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2))
        ### END YOUR CODE
#         return NotImplementedError()
        return matrix

    def most_similar(self, word, num_candidates=3):
        """
        Retrieve most similar words to the desire word.
        Args:
            - word: string.
        Return:
            List of top num_candidates most similar words, e.g., ["man", "woman", "king"].
            Hint: You can compute similarity scores between word and all other words in the vocabulary,
                and select the top num_candidates with largest scores.
                Note: DO NOT include word itself to the return list.
        """
        ## YOUR CODE HERE
        most_similar = []
        if word in self.embeddings.keys():
            dist = {}
            for word2 in self.embeddings.keys():
                if word2 == word:
                    continue
                dist[word2] = self.word_similarity(word, word2)
#                 print(dist[word2])
            sorted_dist = sorted(dist, key=dist.get, reverse=True)
            return [word for i, word in enumerate(sorted_dist) if i<num_candidates]
        return []
        ## END YOUR CODE
#         return NotImplementedError()

    def get_oov_vector(self):
        """Compute the unknown word vector by averaging all known vectors."""
        vectors = [vec for vec in self.embeddings.values()]
        oov_vector = np.mean(vectors, axis=0)
        return oov_vector

    def visualization(self, word_list=["sister", "brother", "man", "woman", "uncle", "aunt"]):
        """Plot words in the plain coordinate.
        Data reduction method PCA, ie, principal component analysis, is used for the visualization.
        Args:
            - word_list: List of words you want to plot.
        """
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        X = []
        for word in word_list:
            vector = self.get_vector(word)
            X.append(vector)

        pca = PCA(n_components=2)
        Y = pca.fit_transform(X)
        plt.scatter(Y[:, 0], Y[:, 1])
        for label, x, y in zip(word_list, Y[:, 0], Y[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
        plt.show()


if __name__ == '__main__':

    # Init from file
    model = PretrainedWord2Vec("/userhome/34/zxzhao/glove.6B.50d.txt")

    #
    print("Vocabulary size", model.vocab_size)
    print("\n")

    model.health_check()
    print("\n")

    # Word vector
    for word in ["queen", "queeeen"]:
        vector = model.get_vector(word)
        print("Vector for %s:" % word)
        print(vector)
    print("\n")

    # Word similarity
    sim = model.word_similarity("king", "queen")
    print("Similarity between `king` and `queen`:")
    print(sim)
    print("\n")

    # Top n similar words
    top_n_words = model.most_similar("queen", 3)
    print("Top 10 similar words to `queen':")
    for w in top_n_words:
        print(w)

    print("\n")

    # Visualization
    model.visualization()
