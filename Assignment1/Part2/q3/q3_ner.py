import tensorflow as tf
import os
from datasets.conll_utils import ConllDataset
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score

tf.random.set_seed(32767)


class Encoder(tf.keras.Model):
    """
    Encoder for NER model.
    Args:
        - vocab_size: vocabulary size, integer.
        - embedding_size: embedding size, integer.
        - enc_units: hidden size of LSTM layer, integer.
    This is a sequence-to-sequence model implementation with Tensorflow. See
    https://www.tensorflow.org/tutorials/text/nmt_with_attention
    for reference.
    """

    def __init__(self, vocab_size, embedding_size, enc_units):
        super(Encoder, self).__init__()

        # Word embedding layer.
        # Hint: use tf.keras.layers.Embedding module.
        ### YOUR CODE HERE (1 line)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        ### END YOUR CODE

        # LSTM layer with units of enc_units
        # Hint: use tf.keras.layers.LSTM module with setting `return_sequences=True'.
        ### YOUR CODE HERE (1 line)
        self.lstm = tf.keras.layers.LSTM(enc_units, return_sequences=True)
        ### END YOUR CODE

    def call(self, x):
        """
        Args:
            - x: Input tensor of shape (batch_size, sequence_length), dtype np.int32
        Return:
            Tensor of shape (batch_size, sequence_length, enc_units), dtype tf.float32
        """
        # Given input tensor x, compute the output tensor after
        # passing through word embedding layer, and LSTM layer.
        ### YOUR CODE HERE (2 lines)
        x = self.embedding(x)
        x = self.lstm(x)
        return x
        ### END YOUR CODE
#         return NotImplementedError()


class FFC(tf.keras.Model):
    """
    Fully-connected feed-forward layer for NER model.
    Args:
        - ffc_units: hidden units of feedforward layer, integer.
        - num_labels: number of named entities. The value should be (actual_num_labels + 1),
            because zero paddings are added to the sequences.
    """
    
    def __init__(self, ffc_units, num_labels):
        super(FFC, self).__init__()

        # Add two fully connected layers.
        # First layer outputs dimensionality of ffc_units
        # Second layer outputs dimensionality of num_labels
        # Hint: use tf.keras.layers.Dense module.

        ### YOUR CODE HERE (2 lines)
        self.layer1 = tf.keras.layers.Dense(ffc_units)
        self.layer2 = tf.keras.layers.Dense(num_labels)
        ### END YOUR CODE

    def call(self, enc_output):
        """
        Args:
            - enc_output: Input tensor of shape (batch_size, sequence_length, enc_units),
                dtype tf.float32
        Return:
            Tensor of shape (batch_size, sequence_length, num_labels), dtype tf.float32
        """
        # Given input tensor enc_output, compute the output tensor
        # after passing through two fully-connected layers.
        ### YOUR CODE HERE (2 lines)
        x = self.layer1(enc_output)
        return self.layer2(x)
        ### END YOUR CODE
#         return NotImplementedError()


class NERModel(tf.keras.Model):
    """NER model is the stack of Encoder and FeedFoward networks."""

    def __init__(self, vocab_size, embedding_size, enc_units, ffc_units, num_labels):
        super(NERModel, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_size, enc_units)
        self.ffc = FFC(ffc_units, num_labels)

    def call(self, x):
        """
        Args:
            - x: Input tensor of shape (batch_size, sequence_length)
        Return:
            Tensor of shape (batch_size, sequence_length, num_labels)
        """
        # Compute final logits to predict the named entity labels.
        enc_output = self.encoder(x)
        logits = self.ffc(enc_output)
        return logits


class MaskedLoss(tf.keras.losses.Loss):
    """Masked loss function"""

    def __init__(self):
        super(MaskedLoss, self).__init__()
        # Use `SparseCategoricalCrossentropy` for multiple classes model.
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def call(self, y_true, logits):
        """
        Args:
            - y_true: True labels of shape (batch_size, sequence_length)
            - logits: Predicted logits of shape (batch_size, sequence_length, num_labels)
        Return:
            Tensor of shape (batch_size,)
        """
        # Compute masked loss here. Compute unmasked loss first, and multiply with mask vector later on.
        # To get mask vector, you can
        #   1. Identify each element in y_true 0 or not, using tf.math.euqal
        #   2. Reverse the above results with tf.math.logical_not
        #   3. Convert boolean values to loss.dtype with tf.cast.

        ### YOUR CODE HERE (~4 lines)
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss_ = self.loss_fn(y_true, logits)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_mean(loss_)
        ### END YOUR CODE
#         return NotImplementedError()


class Config:
    """Model config."""
    def __init__(self):
        self.embedding_size = 256
        self.enc_units = 256
        self.ffc_units = 256
        self.num_labels = 6


def train(n_epochs=10, batch_size=128, learning_rate=0.01):
    """Lets train the model."""
    # Load data.
    dataset = ConllDataset()
    vocab_size = dataset.vocab_size
    x_train = dataset.x_train
    y_train = dataset.y_train
    x_dev = dataset.x_dev
    y_dev = dataset.y_dev

    # Initialize model and loss function.
    config = Config()
    model = NERModel(vocab_size, config.embedding_size, config.enc_units, config.ffc_units, config.num_labels)
    loss_fn = MaskedLoss()

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=loss_fn)

    # Manage checkpoint storage.
    checkpoint_path = "ner_checkpoints/ner-{epoch:02d}.ckpt"
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      save_best_only=True,
                                                      verbose=0)
    # Train the model with train & dev data.
    # The model checkpoints will be saved every epoch.
    # Check `ner_checkpoints' folder.
    model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_dev, y_dev),
              epochs=n_epochs,
              callbacks=[checkpointer])


def predict(checkpoint_path=None, pred_filepath="results/prediction.txt"):
    """Predict named entities with saved checkpoints.
    Args:
        - checkpoint_path: checkpoint path, e.g., `ner_checkpoints/ner-06.ckpt' represents
            the checkpoint saved at 6th epoch. If `None' is given, the latest checkpoint will be loaded.
        - pred_filepath: File path to store the final predicted results.
    """
    if checkpoint_path is None:
        checkpoint_path = tf.train.latest_checkpoint("ner_checkpoints")

    print("Loading model from %s" % checkpoint_path)

    if not os.path.exists("results"):
        os.mkdir("results")

    dataset = ConllDataset()
    vocab_size = dataset.vocab_size
    label2index = dataset.label2index
    index2label = dict([(v, k) for k, v in label2index.items()])
    x_test = dataset.x_test
    testset = dataset.testset

    config = Config()
    embedding_size = config.embedding_size
    enc_units = config.enc_units
    ffc_units = config.ffc_units
    num_labels = config.num_labels

    model = NERModel(vocab_size, embedding_size, enc_units, ffc_units, num_labels)
    model.load_weights(checkpoint_path)
    # Make prediction and convert ids to true labels.
    predictions = model.predict(x_test)
    predicted_indexes = tf.math.argmax(predictions, axis=2)
    assert len(x_test) == len(predicted_indexes)

    predicted_labels = []
    for index, x in zip(predicted_indexes, x_test):
        current_labels = []
        for idx, x_ in zip(index, x):
            if x_ == 0: # Pad
                break
            label = index2label.get(idx.numpy(), "O")
            current_labels.append(label)
        predicted_labels.append(current_labels)

    assert len(testset) == len(predicted_labels)
    # Write to files
    with open(pred_filepath, "w", encoding="utf-8") as fp:
        for (current_tokens, _), pred_labels in zip(testset, predicted_labels):
            assert len(current_tokens) == len(pred_labels)
            for token, label in zip(current_tokens, pred_labels):
                fp.write("{} {}\n".format(token, label))
            fp.write("\n")
    print("Predicted file is saved to {}.".format(pred_filepath))

    print("--------------------------------------------------")
    print("F1-score: ")
    y_test = dataset.y_test
    y_test_true = []
    y_pre = []
    for index, x in zip(y_test, x_test):
        current_labels = []
        for idx, x_ in zip(index, x):
            if x_ == 0: # Pad
                break
            label = index2label.get(idx, "O")
            current_labels += label
        y_test_true += current_labels
    
    for i in predicted_labels:
        y_pre += i
    
#     print(y_test_true[0:10])
#     print(y_pre[0:10])
        
    precision = precision_score(y_test_true, y_pre, average='weighted')
    recall = recall_score(y_test_true, y_pre, average='weighted')
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f1_score)
    return True


if __name__ == '__main__':

#     train(n_epochs=10, batch_size=64, learning_rate=0.0001)
    train(n_epochs=10, batch_size=128, learning_rate=0.01)
    predict()

