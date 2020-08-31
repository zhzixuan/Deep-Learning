import os
import numpy as np
import tensorflow as tf
from q2_word2vec import PretrainedWord2Vec
from datasets.imdb_utils import ImdbDataset, SmallImdbDataset

tf.random.set_seed(394232)


class DataProcessor(object):
    """
    Convert sentence to vector, which is usually called sentence vector.
    The sentence vector is simply computed by averaging all the word vectors in the sentence.
    """

    def __init__(self, glove_path, word2vec_dim=50, debug=True):
        self.word2vec = PretrainedWord2Vec(glove_path=glove_path)
        self.word2vec_dim = word2vec_dim
        self.imdb = SmallImdbDataset() if debug else ImdbDataset()

    def get_single_sentence_vector(self, sentence):
        n_words = len(sentence)
        avg_vec = np.zeros(shape=(self.word2vec_dim,))
        for word in sentence:
            vec = self.word2vec.get_vector(word)
            avg_vec += vec
        avg_vec /= n_words
        return avg_vec

    def get_all_sentence_vectors(self, sentences):
        num_sentences = len(sentences)
        features = np.zeros(shape=[num_sentences, self.word2vec_dim])
        for i in range(num_sentences):
            features[i] = self.get_single_sentence_vector(sentences[i])
        return features

    def load_train_test(self):
        (sent_train, y_train), (sent_test, y_test) = self.imdb.get_train_test()
        x_train = self.get_all_sentence_vectors(sent_train)
        x_test = self.get_all_sentence_vectors(sent_test)
        return (x_train, y_train), (x_test, y_test)


def create_model_and_compile(output_dim=1, hidden_dim=50, learning_rate=0.0001):
    """
    Create and compile tensorflow model.
    Args:
        - learning_rate: float.
        - input_dim: input data dimension, integer.
        - output_dim: model output dimension, integer.
        - hidden_dim: hidden layer dimension, integer.
    Return:
        Tensorflow Sequential object. See
        https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
        for more information.
    """
    # Create two densely-connected neural networks.
    # The first layer output units is hidden_dim and activation is `relu';
    # the second layer output units is output_dim and no activation

    # Hints: (1) use tf.keras.layers.Dense module to define the layers;
    #       and (2) use tf.keras.Sequential to stack them together.

    ### YOUR CODE HERE
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_dim, activation='relu'))
    model.add(tf.keras.layers.Dense(output_dim))
    ### END YOUR CODE

    # Define an Adam optimizer with leaning_rate
    # Hint: Use tf.keras.optimizer module.
    # For more information about Adam optimizer, see
    # https://arxiv.org/abs/1412.6980 for more information.

    ### YOUR CODE HERE
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    ### END YOUR CODE

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train(n_epochs=20, batch_size=32, learning_rate=0.001):
    """
    Train sentiment model.
    Fine-tune n_epochs, batch_size and learning_rate to see how the accuracy is affected.
    Also change activation function from `relu` to `sigmoid` to see how the training goes.
    Args:
        - n_epochs: total training epochs, integer.
        - batch_size: batch size, integer.
        - learning_rate: float. When using Adam optimizer, use relatively smaller learning rate.
    """
    # Use your own glove path.
    # When debug is True, it will use the smaller dataset to train the model.
    processor = DataProcessor(glove_path="/userhome/34/zxzhao/glove.6B.50d.txt", debug=False)
    (x_train, y_train), (x_test, y_test) = processor.load_train_test()

    model = create_model_and_compile(learning_rate=learning_rate)

    print("Start training..\n")
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        epochs=n_epochs
    )
    print("Done.")


if __name__ == '__main__':
    train(n_epochs=20, batch_size=32, learning_rate=1e-4)

