import sys
import numpy
import tensorflow as tf
import io
import pickle
import time


USC_EMAIL = 'azizim@usc.edu'  # TODO(student): Fill to compete on rankings.
PASSWORD = '3e173a6bb4a2a4ce'  # TODO(student): You will be given a password via email.


# class DatasetReader(object):
class DatasetReader():

  # def __init__(self):
  #   self.term_index = {}
  #   self.tag_index = {}

  # TODO(student): You must implement this.
  @staticmethod
  def ReadFile(filename, term_index, tag_index):
  # def ReadFile(self, filename, term_index, tag_index):
    """Reads file into dataset, while populating term_index and tag_index.
   
    Args:
      filename: Path of text file containing sentences and tags. Each line is a
        sentence and each term is followed by "/tag". Note: some terms might
        have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
        separates the tag.
      term_index: dictionary to be populated with every unique term (i.e. before
        the last "/") to point to an integer. All integers must be utilized from
        0 to number of unique terms - 1, without any gaps nor repetitions.
      tag_index: same as term_index, but for tags.

    Return:
      The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
      each parsedLine is a list: [(term1, tag1), (term2, tag2), ...] 
    """

    infile = io.open(filename, 'r', encoding="utf8")
    sentences = infile.readlines()
    infile.close()
    # tags = [[wordtag.rsplit('/', 1)[-1] for wordtag in sentence.strip().split()] for sentence in sentences]
    # words = [[wordtag.rsplit('/', 1)[0] for wordtag in sentence.strip().split()] for sentence in sentences]
    # parsed_file = [[(wordtag.rsplit('/', 1)[0],wordtag.rsplit('/', 1)[-1]) for wordtag in sentence.strip().split()] for sentence in sentences]
    parsed_file = []
    term_index_count = 1
    tag_index_count = 0
    for sentence in sentences:
      parsed_sentence = []
      for wordtag in sentence.strip().split():
        word_tag_split = wordtag.rsplit('/', 1)
        word = word_tag_split[0]
        tag = word_tag_split[-1]
        # parsed_sentence.append((word, tag))
        if word not in term_index.keys():
          term_index[word] = term_index_count
          term_index_count += 1
        if tag not in tag_index.keys():
          tag_index[tag] = tag_index_count
          tag_index_count += 1
        parsed_sentence.append((term_index.get(word), tag_index.get(tag)))
      parsed_file.append(parsed_sentence)
    # self.term_index = term_index
    # self.tag_index = tag_index
    return parsed_file


  # TODO(student): You must implement this.
  @staticmethod
  def BuildMatrices(dataset):
  # def BuildMatrices(self, dataset):
    """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.

    Args:
      dataset: Returned by method ReadFile. It is a list (length N) of lists:
        [sentence1, sentence2, ...], where every sentence is a list:
        [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.

    Returns:
      Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
        terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
          indices in dataset[i].
        tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
          indices in dataset[i].
        lengths: shape (N) int64 numpy array. Entry i contains the length of
          sentence in dataset[i].

      T is the maximum length. For example, calling as:
        BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
      i.e. with two sentences, first with length 2 and second with length 4,
      should return the tuple:
      (
        [[1, 4, 0, 0],    # Note: 0 padding.
         [13, 3, 7, 3]],

        [[2, 10, 0, 0],   # Note: 0 padding.
         [20, 6, 8, 20]], 

        [2, 4]
      )
    """

    # term_index = self.term_index
    # tag_index = self.tag_index
    lengths_arr = numpy.array(list(map(len, dataset)))
    T = max(lengths_arr)
    N = lengths_arr.shape[0]
    terms_matrix = numpy.zeros((N, T))
    tags_matrix = numpy.zeros((N, T))
    for sen_counter, sentence in enumerate(dataset):
      for (word_counter, (word_idx, tag_idx)) in enumerate(sentence):
        terms_matrix[sen_counter, word_counter] = word_idx
        tags_matrix[sen_counter, word_counter] = tag_idx

    return terms_matrix, tags_matrix, lengths_arr


  @staticmethod
  def ReadData(train_filename, test_filename=None):
  # def ReadData(self, train_filename, test_filename=None):
    """Returns numpy arrays and indices for train (and optionally test) data.

    NOTE: Please do not change this method. The grader will use an identitical
    copy of this method (if you change this, your offline testing will no longer
    match the grader).

    Args:
      train_filename: .txt path containing training data, one line per sentence.
        The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
      test_filename: Optional .txt path containing test data.

    Returns:
      A tuple of 3-elements or 4-elements, the later iff test_filename is given.
      The first 2 elements are term_index and tag_index, which are dictionaries,
      respectively, from term to integer ID and from tag to integer ID. The int
      IDs are used in the numpy matrices.
      The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
        - train_terms: numpy int matrix.
        - train_tags: numpy int matrix.
        - train_lengths: numpy int vector.
        These 3 are identical to what is returned by BuildMatrices().
      The 4th element is a tuple of 3 elements as above, but the data is
      extracted from test_filename.
    """

    term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
    tag_index = {}
    # self.term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
    # self.tag_index = {}
    
    train_data = DatasetReader.ReadFile(filename=train_filename, term_index=term_index, tag_index=tag_index)
    train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(dataset=train_data)
    
    if test_filename:
      test_data = DatasetReader.ReadFile(filename=test_filename, term_index=term_index, tag_index=tag_index)
      test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

      if test_tags.shape[1] < train_tags.shape[1]:
        diff = train_tags.shape[1] - test_tags.shape[1]
        zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
        test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
        test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
      elif test_tags.shape[1] > train_tags.shape[1]:
        diff = test_tags.shape[1] - train_tags.shape[1]
        zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
        train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
        train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

      return (term_index, tag_index,
              (train_terms, train_tags, train_lengths),
              (test_terms, test_tags, test_lengths))
    else:
      return term_index, tag_index, (train_terms, train_tags, train_lengths)


class SequenceModel(object):

    def __init__(self, max_length=310, num_terms=1000, num_tags=40):
        """Constructor. You can add code but do not remove any code.

        The arguments are arbitrary: when you are training on your own, PLEASE set
        them to the correct values (e.g. from main()).

        Args:
          max_lengths: maximum possible sentence length.
          num_terms: the vocabulary size (number of terms).
          num_tags: the size of the output space (number of tags).

        You will be passed these arguments by the grader script.
        """
        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags
        self.x = tf.placeholder(tf.int64, [None, self.max_length], 'X')
        self.lengths = tf.placeholder(tf.int32, [None], 'lengths')
        self.targets = tf.placeholder(tf.int64, [None], 'targets')
        self.cell_type = 'rnn'  # 'lstm'
        self.log_step = 50
        self.sess = tf.Session()
        self.size_embed = 15  # HYP
        self.state_size = 10  # HYP
        self.embed = tf.get_variable('embed', shape=[self.num_terms, self.size_embed],
                                     dtype=tf.float32, initializer=None, trainable=True)
        self._accuracy()


    # TODO(student): You must implement this.
    def lengths_vector_to_binary_matrix(self, length_vector):

        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.

        Specifically, the return matrix B will have the following:
          B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """
        num_batch_ = length_vector.shape[0].value
        if num_batch_ == None:
            self.b = tf.placeholder(tf.float32, [None, self.max_length], 'b')
            b = self.b
        elif num_batch_ > 0:
            b = numpy.zeros((num_batch_, self.max_length))
            for i in range(num_batch_):
                b[i, :length_vector[i]] = 1
            b = tf.convert_to_tensor(b, dtype=tf.float32)
        return b

        # return tf.ones([tf.shape(length_vector), self.max_length], dtype=tf.float32)

    # TODO(student): You must implement this.
    def save_model(self, filename):
        """Saves model to a file."""
        # import pickle
        # sess = tf.Session()
        var_dict = {v.name: v for v in tf.global_variables()}
        pickle.dump(self.sess.run(var_dict), open(filename, 'w'))
        return

  # TODO(student): You must implement this.
    def load_model(self, filename):
        """Loads model from a file."""
        # import pickle
        # sess = tf.Session()
        var_values = pickle.load(open(filename))
        assign_ops = [v.assign(var_values[v.name]) for v in tf.global_variables()]
        self.sess.run(assign_ops)
        return

    # TODO(student): You must implement this.
    def build_inference(self):
        """Build the expression from (self.x, self.lengths) to (self.logits).

        Please do not change or override self.x nor self.lengths in this function.

        Hint:
          - Use lengths_vector_to_binary_matrix
          - You might use tf.reshape, tf.cast, and/or tensor broadcasting.
        """
        # TODO(student): make logits an RNN on x.
        self.lens_to_bin = self.lengths_vector_to_binary_matrix(self.lengths)
        # terms_batch = tf.placeholder(tf.int32, shape=[None, None]) #####
        xemb_ = tf.nn.embedding_lookup(params=self.embed, ids=self.x, partition_strategy='mod', name=None,
                                      validate_indices=True,max_norm=None)
        if self.cell_type == 'rnn':
            rnn_cell = tf.keras.layers.SimpleRNNCell(self.state_size, activation='tanh', use_bias=True,
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='orthogonal',recurrent_dropout=0.0,
                                                     bias_initializer='zeros',kernel_regularizer=None,
                                                     recurrent_regularizer=None,bias_regularizer=None,
                                                     kernel_constraint=None,recurrent_constraint=None,
                                                     bias_constraint=None, dropout=0.0)
        elif self.cell_type == 'lstm':
            rnn_cell = tf.keras.layers.LSTMCell__init__(units=self.state_size, activation='tanh',
                                                        recurrent_activation='hard_sigmoid', use_bias=True,
                                                        kernel_initializer='glorot_uniform',
                                                        recurrent_initializer='orthogonal', bias_initializer='zeros',
                                                        unit_forget_bias=True,kernel_regularizer=None,
                                                        recurrent_regularizer=None, bias_regularizer=None,
                                                        kernel_constraint=None,recurrent_constraint=None,
                                                        bias_constraint=None, dropout=0.0,
                                                        recurrent_dropout=0.0, implementation=1)
        else:
            # cell_fw = tf.contrib.rnn.LSTMCell(self.state_size, state_is_tuple=True)
            # cell_bw = tf.contrib.rnn.LSTMCell(self.state_size, state_is_tuple=True)
            # _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
            #                                                                       cell_bw, char_embeddings,
            #                                                                       sequence_length=word_lengths,
            #                                                                       dtype=tf.float32)
            print("Wrong cell type")

        states = []
        cur_state = tf.zeros(shape=[1, self.state_size])

        # 2. put the time dimension on axis=1 for dynamic_rnn
        s = tf.shape(xemb_)  # store old shape
        # shape = (batch x sentence, word, dim of char embeddings)
        # xemb = tf.reshape(xemb_, shape=[-1, s[-2], s[-1]])  # (batch_size, timesteps, features)
        xemb = xemb_
        # word_lengths = tf.reshape(self.word_lengths, shape=[-1])


        for i in range(self.max_length):
            cur_state = rnn_cell(xemb[:, i, :], [cur_state])[0]  # shape (batch, state_size)
            states.append(cur_state)
        stacked_states = tf.stack(states, axis=1)  # Shape (batch, max_length, state_size)
        # logits: A Tensor of shape[batch_size, sequence_length, num_decoder_symbols] and dtype float.
        self.logits = tf.contrib.layers.fully_connected(stacked_states, self.num_tags, activation_fn=tf.nn.softmax,
                                                   normalizer_fn=None, normalizer_params=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   weights_regularizer=None, biases_initializer=tf.zeros_initializer(),
                                                   biases_regularizer=None, reuse=None, variables_collections=None,
                                                   outputs_collections=None, trainable=True, scope=None)
        return self.logits

    # TODO(student): You must implement this.
    def run_inference(self, terms, lengths):
        """Evaluates self.logits given self.x and self.lengths.

        Hint: This function is straight forward and you might find this code useful:
        # logits = session.run(self.logits, {self.x: tags, self.lengths: lengths})
        # return numpy.argmax(logits, axis=2)

        Args:
          tags: numpy int matrix, like terms_matrix made by BuildMatrices.
          lengths: numpy int vector, like lengths made by BuildMatrices.

        Returns:
          numpy int matrix of the predicted tags, with shape identical to the int
          matrix tags i.e. each term must have its associated tag. The caller will
          *not* process the output tags beyond the sentence length i.e. you can have
          arbitrary values beyond length.
        """
        # logits = self.sess.run(self.logits, {self.x: terms, self.lengths: lengths})
        logits = self.build_inference()
        return tf.argmax(logits, axis=2)
        # return numpy.zeros_like(tags)

    # TODO(student): You must implement this.
    def build_training(self):
        """Prepares the class for training.

        It is up to you how you implement this function, as long as train_on_batch
        works.

        Hint:
          - Lookup tf.contrib.seq2seq.sequence_loss
          - tf.losses.get_total_loss() should return a valid tensor (without raising
            an exception). Equivalently, tf.losses.get_losses() should return a
            non-empty list.
        """
        self.targets = tf.placeholder(tf.int32, shape=[None, self.max_length], name="targets")
        # logits: A Tensor of shape [batch_size, sequence_length, num_decoder_symbols] and dtype float.
        # targets: A Tensor of shape[batch_size, sequence_length] and dtype int.
        # weights: A Tensor of shape[batch_size, sequence_length] and dtype float
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.targets,
                                                     weights=self.lens_to_bin, average_across_timesteps=True,
                                                     average_across_batch=True, softmax_loss_function=None, name=None)
        opt = tf.train.AdamOptimizer() #HYP
        self.train_op = opt.minimize(self.loss)
        # print(tf.losses.get_total_loss(add_regularization_losses=True,
        #                                name='total_loss'))  # should return a valid tensor
        # print(tf.losses.get_losses())  # should return anon-empty list
        return

    def _accuracy(self):
        predict = self.run_inference(self.x, self.lengths)
        correct = tf.equal(predict, self.targets)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
        return self.accuracy_op

    def train_epoch(self, terms, tags, lengths, batch_size=32, learn_rate=1e-7):
        #HYP
        """Performs updates on the model given training training data.

        This will be called with numpy arrays similar to the ones created in
        Args:
          terms: int64 numpy array of size (# sentences, max sentence length)
          tags: int64 numpy array of size (# sentences, max sentence length)
          lengths:
          batch_size: int indicating batch size. Grader script will not pass this,
            but it is only here so that you can experiment with a "good batch size"
            from your main block.
          learn_rate: float for learning rate. Grader script will not pass this,
            but it is only here so that you can experiment with a "good learn rate"
            from your main block.
        """
        self.batch_size = batch_size
        step = 0
        losses = []
        accuracies = []
        num_training = len(terms)
        self.sess.run(tf.global_variables_initializer())
        for i in range(num_training // batch_size):
            x_batch = terms[i * batch_size:(i + 1) * batch_size][:]
            tags_batch = tags[i * batch_size:(i + 1) * batch_size]
            lengths_batch = lengths[i * batch_size:(i + 1) * batch_size]
            feed_dict = {self.x: x_batch, self.lengths: lengths_batch,
                         self.targets: tags_batch,
                         self.b: numpy.repeat(lengths_batch.reshape(self.batch_size, 1), self.max_length, axis=1)}
            fetches = [self.train_op, self.loss, self.accuracy_op]
            _, loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
            losses.append(loss)
            accuracies.append(accuracy)

            if step % self.log_step == 0:
                print('iteration (%d): train batch loss = %.3f, train batch accuracy = %.3f' %
                      (step, loss, accuracy))
            step += 1

        # plt.title('Training loss')
        # loss_hist_ = losses[1::100]  # sparse the curve a bit
        # plt.plot(loss_hist_, '-o')
        # plt.xlabel('epoch')
        # plt.gcf().set_size_inches(15, 12)
        # plt.show()
        return

    # TODO(student): You can implement this to help you, but we will not call it.
    def evaluate(self, terms, tags, lengths):
        feed_dict = {self.x: terms, self.lengths: lengths,
                     self.targets: tags}
        fetches = [self.train_op, self.loss, self.accuracy_op]
        _, loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
        print('accuracy on test: {}'.format(accuracy))


def main():
    """This will never be called by us, but you are encouraged to implement it for
    local debugging e.g. to get a good model and good hyper-parameters (learning
    rate, batch size, etc)."""
    # Read dataset.
    reader = DatasetReader()
    # train_filename = sys.argv[1]
    # train_filename = "F:\Acad\Spring19\CSCI544_NLP\code_hw\HW3\HW_data\ja_gsd_train_tagged.txt"  # japonease
    train_filename = "F:\Acad\Spring19\CSCI544_NLP\code_hw\HW3\HW_data\it_isdt_train_tagged.txt"
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename=train_filename, test_filename=test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data

    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    time0 = time.time()
    K = 300
    epoch = 0
    print('-' * 5 + '  Start training  ' + '-' * 5)
    while time.time()-time0 <= K:
        print("train epoch {}".format(epoch+1))
        model.train_epoch(train_terms, train_tags, train_lengths)
        print('Finished epoch %i. Evaluating ...' % (epoch + 1))
        model.evaluate(test_terms, test_tags, test_lengths)
        epoch += 1

if __name__ == '__main__':
    main()