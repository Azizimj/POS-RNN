import sys
import numpy
import tensorflow as tf
import io
import pickle
import time

tf.random.set_random_seed(110)
# seed(110)

USC_EMAIL = 'azizim@usc.edu'  # TODO(student): Fill to compete on rankings.
PASSWORD = '3e173a6bb4a2a4ce'  # TODO(student): You will be given a password via email.

# batch_size = 10

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

    infile = open(filename, 'r', encoding="utf8")
    sentences = infile.readlines()
    infile.close()
    # tags = [[wordtag.rsplit('/', 1)[-1] for wordtag in sentence.strip().split()] for sentence in sentences]
    # words = [[wordtag.rsplit('/', 1)[0] for wordtag in sentence.strip().split()] for sentence in sentences]
    # parsed_file = [[(wordtag.rsplit('/', 1)[0],wordtag.rsplit('/', 1)[-1]) for wordtag in sentence.strip().split()] for sentence in sentences]
    parsed_file = []
    term_index_count = int(len(term_index))
    tag_index_count = int(len(tag_index))
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

    terms_matrix = numpy.array(terms_matrix).astype(int)
    tags_matrix = numpy.array(tags_matrix).astype(int)
    lengths_arr = numpy.array(lengths_arr).astype(int)
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
        self.x = tf.placeholder(tf.int32, [None, self.max_length], 'X')
        self.lengths = tf.placeholder(tf.int64, [None], 'lengths')
        self.tags = tf.placeholder(tf.int64, [None, self.max_length], 'tags')
        # I usually prefer int32 for space and speed, but the embedding_lookup function expects int64
        # self.cell_type = 'rnn'
        self.cell_type = 'lstm'
        # self.cell_type = 'bidic_rnn'
        # self.cell_type = 'bidic_lstm'
        self.log_step = 10
        self.sess = tf.Session()
        self.size_embed = 40  # HYP
        self.state_size = 15  # HYP
        self.b = tf.placeholder(tf.float32, [None, self.max_length], 'b')
        self.learn_rate = tf.placeholder(tf.float32, [], 'lr')
        self.dropout_keep_prob = 1  #HYP
        self.use_fc = True
        self.epoch_return = True
        self.use_bn = True
        print("size_embed {}, state_size {}, dropout_keep_prob {}, use_fc {}, epoch_return {} usc_bn {}".format(
            self.size_embed, self.state_size, self.dropout_keep_prob, self.use_fc, self.epoch_return, self.use_bn
        ))
        # self._accuracy()


    # TODO(student): You must implement this.
    def lengths_vector_to_binary_matrix(self, length_vector):

        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.

        Specifically, the return matrix B will have the following:
          B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """
        # num_batch_ = length_vector.shape[0].value
        # if num_batch_ == None:
        self.lens_to_bin = self.b

        # if self.is_build:
        #     self.lens_to_bin = self.b
        # else:
        #     num_batch_ = len(length_vector)
        #     self.lens_to_bin = numpy.zeros((num_batch_, self.max_length))
        #     for i in range(num_batch_):
        #         self.lens_to_bin[i, :length_vector[i]] = 1
        #     self.lens_to_bin = tf.convert_to_tensor(self.lens_to_bin, dtype=tf.float32)

        # lengths is a placeholder.Your task here is to use it to make a binary matrix.For this, you might
        # find the following useful:
        # TensorFlow broadcasting[automatic, google for it].tf.expand_dims, tf.range, casting, and comparator
        # operators. Or, you can do while -loops in TensorFlow, though if I was programming, I would look for
        # a mathematical expression i.e.the functions above.
        len_to_bin_f = lambda x: tf.concat([tf.broadcast_to(1, [1, x]),tf.broadcast_to(0, [1, self.max_length - x])], 1)
        a = tf.map_fn(len_to_bin_f, length_vector)
        # for i in tf.range(length_vector.shape[0]):
            # b[i] =



        return self.lens_to_bin

        # return tf.ones([tf.shape(length_vector), self.max_length], dtype=tf.float32)

    # TODO(student): You must implement this.
    def save_model(self, filename):
        """Saves model to a file."""
        import pickle
        # sess = tf.Session()
        var_dict = {v.name: v for v in tf.global_variables()}
        pickle.dump(self.sess.run(var_dict), open(filename, 'bw'))
        # saver = tf.train.Saver()
        # model_path = saver.save(self.sess, "model.ckpt")
        return

  # TODO(student): You must implement this.
    def load_model(self, filename):
        """Loads model from a file."""
        import pickle
        # tf.reset_default_graph()
        # tf.global_variables_initializer()
        # variables = tf.global_variables()
        self.sess = tf.Session()
        var_values = pickle.load(open(filename, 'rb'))
        assign_ops = [v.assign(var_values[v.name]) for v in tf.global_variables()]
        self.sess.run(assign_ops)
        # param_dict = {}
        # for var in variables:
        #     var_name = var.name[:-2]
        #     # print('Loading {} from checkpoint. Name: {}'.format(var.name, var_name))
        #     param_dict[var_name] = var
        # saver = tf.train.Saver()
        # saver.restore(self.sess, "model.ckpt")
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
        # tf.reset_default_graph()
        # if 'embed:0' not in [v.name for v in tf.global_variables()]:
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            self.embed = tf.get_variable('embed', shape=[self.num_terms, self.size_embed],
                                         dtype=tf.float32, initializer=None, trainable=True)

            self.lens_to_bin = self.lengths_vector_to_binary_matrix(self.lengths)
            # terms_batch = tf.placeholder(tf.int32, shape=[None, None]) #####
            xemb_ = tf.nn.embedding_lookup(params=self.embed, ids=self.x, partition_strategy='mod', name=None,
                                          validate_indices=True, max_norm=None)
            states = []
            # if self.use_fc == True:
            #     cur_state = tf.zeros(shape=[1, self.state_size])
            # else:
            #     cur_state = tf.zeros(shape=[1, self.num_tags])
            #     self.state_size = self.num_tags

            if not self.use_fc:
                self.state_size = self.num_tags

            # 2. put the time dimension on axis=1 for dynamic_rnn
            s = tf.shape(xemb_)  # store old shape
            # shape = (batch x sentence, word, dim of char embeddings)
            # xemb = tf.reshape(xemb_, shape=[-1, s[-2], s[-1]])  # (batch_size, timesteps, features)
            xemb = xemb_
            # word_lengths = tf.reshape(self.word_lengths, shape=[-1])

            if self.cell_type == 'rnn':
                # rnn_cell = tf.keras.layers.SimpleRNNCell(self.state_size, activation='tanh', use_bias=True,
                #                                          kernel_initializer='glorot_uniform',
                #                                          recurrent_initializer='orthogonal',recurrent_dropout=0.0,
                #                                          bias_initializer='zeros',kernel_regularizer=None,
                #                                          recurrent_regularizer=None,bias_regularizer=None,
                #                                          kernel_constraint=None,recurrent_constraint=None,
                #                                          bias_constraint=None, dropout=0.0)
                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
                if self.dropout_keep_prob is not None:
                    rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob,
                                                             input_keep_prob=1.0,
                                                             state_keep_prob=1.0)
                # for i in range(self.max_length):
                #     cur_state = rnn_cell(xemb[:, i, :], [cur_state])[0]  # shape (batch, state_size)
                #     states.append(cur_state)
                # stacked_states = tf.stack(states, axis=1)  # Shape (batch, max_length, state_size)

                stacked_states = tf.nn.dynamic_rnn(rnn_cell, xemb, dtype=tf.float32)[0]
                if self.use_bn:
                    stacked_states = tf.keras.layers.BatchNormalization()(stacked_states)
                # rnn_cell = tf.keras.layers.RNN(rnn_cell, return_sequences=False, return_state=False,
                #                                     go_backwards =False, stateful=False, unroll=False,
                #                                     time_major =False)
                # stacked_states = rnn_cell(xemb)
            elif self.cell_type == 'lstm':
                # rnn_cell = tf.keras.layers.LSTMCell(units=self.state_size, activation='tanh')
                # rnn_cell = tf.nn.rnn_cell.LSTMCell(self.state_size,)
                # rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_size,
                #                                         forget_bias=1.0, state_is_tuple=True,
                #                                         activation=None, reuse=None, name=None, dtype=None)
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_size, reuse=tf.AUTO_REUSE)
                if self.dropout_keep_prob is not None:
                    rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob,
                                                             input_keep_prob=1.0,
                                                             state_keep_prob=1.0)
                # rnn_cell = tf.keras.layers.LSTMCell(units=self.state_size, activation='tanh',
                #                                     recurrent_activation='hard_sigmoid', use_bias=True,
                #                                     kernel_initializer='glorot_uniform',
                #                                     recurrent_initializer='orthogonal', bias_initializer='zeros',
                #                                     unit_forget_bias=True, kernel_regularizer=None,
                #                                     recurrent_regularizer=None, bias_regularizer=None,
                #                                     kernel_constraint=None, recurrent_constraint=None,
                #                                     bias_constraint=None, dropout=0.0,
                #                                     recurrent_dropout=0.0, implementation=1)
                stacked_states = tf.nn.dynamic_rnn(rnn_cell, inputs=xemb, dtype=tf.float32)[0]
                if self.use_bn:
                    stacked_states = tf.keras.layers.BatchNormalization()(stacked_states)
            elif self.cell_type == "bidic_rnn":
                rnn_fw_cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size, reuse=tf.AUTO_REUSE)  # forward direction cell
                rnn_bw_cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size, reuse=tf.AUTO_REUSE)  # backward direction cell
                if self.dropout_keep_prob is not None:
                    rnn_fw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_cell, output_keep_prob=self.dropout_keep_prob,
                                                                input_keep_prob=1.0,
                                                                state_keep_prob=1.0)
                    rnn_bw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_cell, output_keep_prob=self.dropout_keep_prob,
                                                                input_keep_prob=1.0,
                                                                state_keep_prob=1.0)
                # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
                #                            output: A tuple (outputs, output_states)
                #                                    where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
                stacked_states = tf.concat(tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell, rnn_bw_cell, xemb,
                                                                           dtype=tf.float32)[0], axis=2)  # [batch_size,sequence_length,hidden_size*2]
            elif self.cell_type == 'bidic_lstm':
                lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size,reuse=tf.AUTO_REUSE)  # forward direction cell
                lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size,reuse=tf.AUTO_REUSE)  # backward direction cell
                if self.dropout_keep_prob is not None:
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob,
                                                                 input_keep_prob=1.0,
                                                                 state_keep_prob=1.0)
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob,
                                                                 input_keep_prob=1.0,
                                                                 state_keep_prob=1.0)
                # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
                #                            output: A tuple (outputs, output_states)
                #                                    where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
                stacked_states = tf.concat(tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, xemb,
                                                             dtype=tf.float32)[0], axis=2) #[batch_size,sequence_length,hidden_size*2]
            else:
                # cell_fw = tf.contrib.rnn.LSTMCell(self.state_size, state_is_tuple=True)
                # cell_bw = tf.contrib.rnn.LSTMCell(self.state_size, state_is_tuple=True)
                # _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                #                                                                       cell_bw, char_embeddings,
                #                                                                       sequence_length=word_lengths,
                #                                                                       dtype=tf.float32)
                stacked_states = 0
                print("Wrong cell type")

            # logits: A Tensor of shape[batch_size, sequence_length, num_decoder_symbols] and dtype float.
            if self.use_fc == True:
                self.logits = tf.contrib.layers.fully_connected(stacked_states, self.num_tags, activation_fn=tf.nn.softmax,
                                                           normalizer_fn=None, normalizer_params=None,
                                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                           weights_regularizer=None, biases_initializer=tf.zeros_initializer(),
                                                           biases_regularizer=None, reuse=None, variables_collections=None,
                                                           outputs_collections=None, trainable=True, scope=None)
            else:
                self.logits = stacked_states
                # self.logits = tf.cast(self.logits, dtype=tf.int32)

            self._accuracy()
            self.build_training()
        return self.logits

    # TODO(student): You must implement this.
    def run_inference_(self, terms, lengths):
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
        # logits = self.build_inference()
        logits = self.logits
        # return tf.cast(tf.argmax(logits, axis=2), tf.int64)
        return tf.argmax(logits, axis=2)
        # return numpy.zeros_like(tags)

    def run_inference(self, terms, lengths):
        """Evaluates self.logits given self.x and self.lengths.

        Hint: This function is straight forward and you might find this code useful:
        # logits = session.run(self.logits, {self.x: terms, self.lengths: lengths})
        # return numpy.argmax(logits, axis=2)

        Args:
            terms: numpy int matrix, like terms_matrix made by BuildMatrices.
            lengths: numpy int vector, like lengths made by BuildMatrices.

        Returns:
            numpy int matrix of the predicted tags, with shape identical to the int
            matrix tags i.e. each term must have its associated tag. The caller will
            *not* process the output tags beyond the sentence length i.e. you can have
            arbitrary values beyond length.
        """
        logits = self.sess.run(self.logits, {self.x: terms, self.lengths: lengths})
        return numpy.argmax(logits, axis=2)

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
        # logits: A Tensor of shape [batch_size, sequence_length, num_decoder_symbols] and dtype float.
        # targets: A Tensor of shape[batch_size, sequence_length] and dtype int.
        # weights: A Tensor of shape[batch_size, sequence_length] and dtype float
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.tags,
                                                     weights=self.lens_to_bin, average_across_timesteps=True,
                                                     average_across_batch=True, softmax_loss_function=None, name=None)
        tf.losses.add_loss(self.loss, loss_collection=tf.GraphKeys.LOSSES)
        # g_s = tf.Variable(0, trainable=False)
        # l_r = tf.train.exponential_decay(self.learn_rate, g_s, 500, .96, staircase=True)
        # l_r = self.learn_rate
        l_r = 1e-2
        opt = tf.train.AdamOptimizer(learning_rate=l_r) #HYP
        # opt = tf.train.AdamOptimizer()  # HYP
        # opt = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
        self.train_op = opt.minimize(self.loss, var_list=tf.trainable_variables())
        print('tf.losses.get_total_loss', tf.losses.get_total_loss(add_regularization_losses=True,
                                       name='total_loss'))  # should return a valid tensor
        print('tf.losses.get_losses', tf.losses.get_losses())  # should return a non-empty list
        self.sess.run(tf.global_variables_initializer())
        return

    def _accuracy(self):
        self.predict = self.run_inference_(self.x, self.lengths)
        self.lens_to_bin = self.lengths_vector_to_binary_matrix(self.lengths)
        self.correct = tf.multiply(tf.cast(tf.equal(self.predict, self.tags), tf.float32), self.lens_to_bin)
        # self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
        self.accuracy_op = tf.divide(tf.reduce_sum(self.correct), tf.cast(tf.reduce_sum(self.lengths), tf.float32))
        return self.accuracy_op

    def lengths_to_binary(self, length_vector):
        num_batch_ = len(length_vector)
        b_ = numpy.zeros((num_batch_, self.max_length))
        for i in range(num_batch_):
            b_[i, :length_vector[i]] = 1
        return b_.astype(float)

    def train_epoch(self, terms, tags, lengths, batch_size=10, learn_rate=1e-7):
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
        # self.learn_rate = learn_rate
        self.batch_size = batch_size
        step = 0
        losses = []
        accuracies = []
        num_training = len(terms)
        for i in range(num_training // batch_size):
        # for i in range(50):
            x_batch = terms[i * batch_size:(i + 1) * batch_size][:]
            tags_batch = tags[i * batch_size:(i + 1) * batch_size]
            lengths_batch = lengths[i * batch_size:(i + 1) * batch_size]
            # x_batch = terms
            # tags_batch = tags
            # lengths_batch = lengths
            b_ = self.lengths_to_binary(lengths_batch)
            feed_dict = {self.x: x_batch, self.lengths: lengths_batch,
                         self.tags: tags_batch.astype(numpy.int64),
                         self.b: b_, self.learn_rate: learn_rate}
            fetches = [self.train_op, self.loss, self.accuracy_op, self.correct, self.lengths, self.logits, self.predict]
            # fetches = [self.loss, self.accuracy_op, self.correct, self.lengths, self.logits,
            #            self.predict]
            _, loss, accuracy, correct, lens, logits_, predicts = self.sess.run(fetches, feed_dict=feed_dict)
            losses.append(loss)
            accuracies.append(accuracy)

            if step % self.log_step == 0:
                train_acc = self.evaluate(terms, tags, lengths, self.batch_size)
                print('iteration (%d)/(%d): train batch loss = %.3f, train batch accuracy = %.3f, train acc= %.3f' %
                      (step, num_training // batch_size, loss, accuracy, train_acc))
            step += 1
        # Finally, make sure you uncomment the `return True` below.
        return self.epoch_return

    # TODO(student): You can implement this to help you, but we will not call it.
    def evaluate(self, terms, tags, lengths, batch_size):
        eval_accuracy = 0.0
        eval_iter = 0
        self.batch_size = batch_size
        for i in range(terms.shape[0] // self.batch_size):
            x_batch = terms[i * self.batch_size:(i + 1) * self.batch_size][:]
            tags_batch = tags[i * self.batch_size:(i + 1) * self.batch_size]
            lengths_batch = lengths[i * self.batch_size:(i + 1) * self.batch_size]
            b_ = self.lengths_to_binary(lengths_batch)
            feed_dict = {self.x: x_batch, self.lengths: lengths_batch,
                         self.tags: tags_batch.astype(numpy.int64),
                         self.b: b_, self.learn_rate: 1}
            fetches = [self.accuracy_op, self.predict]
            accuracy, predict = self.sess.run(fetches, feed_dict=feed_dict)
            eval_accuracy += accuracy
            eval_iter += 1
        print('accuracy on val: {}'.format(eval_accuracy / eval_iter))
        return eval_accuracy / eval_iter


def main():
    """This will never be called by us, but you are encouraged to implement it for
    local debugging e.g. to get a good model and good hyper-parameters (learning
    rate, batch size, etc)."""
    # Read dataset.
    reader = DatasetReader()
    # train_filename = sys.argv[1]
    # train_filename = "F:\Acad\Spring19\CSCI544_NLP\code_hw\HW3\HW_data\ja_gsd_train_tagged.txt"  # japonease
    train_filename = "F:\Acad\Spring19\CSCI544_NLP\code_hw\HW3\HW_data\ja_gsd_train_tagged_small.txt"  # japonease
    # train_filename = "F:\Acad\Spring19\CSCI544_NLP\code_hw\HW3\HW_data\it_isdt_train_tagged.txt"
    # train_filename = "F:\Acad\Spring19\CSCI544_NLP\code_hw\HW3\HW_data\it_isdt_train_tagged_small.txt"
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename=train_filename, test_filename=test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    # (train_terms, train_tags, train_lengths) = (train_terms[:5], train_tags[:5], train_lengths[:5])
    (test_terms, test_tags, test_lengths) = test_data

    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    time0 = time.time()
    K = 1
    epoch = 0
    eval_batch_size = 10
    best_val_acc = 0
    best_val_acc_epoch = 0
    print('-' * 5 + '  Start training  ' + '-' * 5)
    # sess = model.sess
    # sess.run(tf.global_variables_initializer())
    while time.time()-time0 <= K:
        print("train epoch {}".format(epoch+1))
        model.train_epoch(train_terms, train_tags, train_lengths, batch_size=20, learn_rate=1e-5)
        print('Finished epoch %i. Evaluating ...' % (epoch + 1))
        tmp_val_acc = model.evaluate(test_terms, test_tags, test_lengths, eval_batch_size)
        if tmp_val_acc > best_val_acc:
            best_val_acc = tmp_val_acc
            best_val_acc_epoch = epoch
        epoch += 1
    print('bets val acc {} in epoch {}'.format(best_val_acc, best_val_acc_epoch))
    model.save_model('model.pkl')
    # import IPython; IPython.embed()
    tf.reset_default_graph()
    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.load_model('model.pkl')
    model.run_inference(test_terms, test_lengths)
    model.evaluate(test_terms, test_tags, test_lengths, eval_batch_size)

    print('time {}'.format(time.time()-time0))

if __name__ == '__main__':
    main()