"""The text-based ideal point model (TBIP) using Tensorflow 2.0.

Supports training either with model.fit() or iterative training."""
import os
import time

from absl import app
from absl import flags
import numpy as np
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp


flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Adam learning rate.")
flags.DEFINE_integer("max_steps",
                     default=1000000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("num_topics",
                     default=50,
                     help="Number of topics.")
flags.DEFINE_integer("batch_size",
                     default=512,
                     help="Batch size.")
flags.DEFINE_integer("num_samples",
                     default=1,
                     help="Number of samples to use for ELBO approximation.")
flags.DEFINE_enum("counts_transformation",
                  default="nothing",
                  enum_values=["nothing", "binary", "sqrt", "log"],
                  help="Transformation used on counts data.")
flags.DEFINE_boolean("pre_initialize_parameters",
                     default=True,
                     help="Whether to use pre-initialized document and topic "
                          "intensities (with Poisson factorization).")
flags.DEFINE_boolean("train_iteratively",
                     default=True,
                     help="Whether to train iteratively or with model.fit().")
flags.DEFINE_string("data",
                    default="senate-speeches-114",
                    help="Data source being used.")
flags.DEFINE_integer("senate_session",
                     default=113,
                     help="Senate session (used only when data is "
                          "'senate-speech-comparisons'.")
flags.DEFINE_integer("print_steps",
                     default=500,
                     help="Number of steps to print and save results.")
flags.DEFINE_integer("seed",
                     default=123,
                     help="Random seed to be used.")
FLAGS = flags.FLAGS


def build_input_pipeline(data_dir, 
                         batch_size, 
                         random_state, 
                         train_iteratively,
                         counts_transformation="nothing"):
  """Load data and build iterator for minibatches.
  
  Args:
    data_dir: The directory where the data is located. There must be four
      files inside the rep: `counts.npz`, `author_indices.npy`, 
      `author_map.txt`, and `vocabulary.txt`.
    batch_size: The batch size to use for training.
    random_state: A NumPy `RandomState` object, used to shuffle the data.
    counts_transformation: A string indicating how to transform the counts.
      One of "nothing", "binary", "log", or "sqrt".
  """
  counts = sparse.load_npz(os.path.join(data_dir, "counts.npz"))
  num_documents, num_words = counts.shape
  author_indices = np.load(
      os.path.join(data_dir, "author_indices.npy")).astype(np.int32) 
  num_authors = np.max(author_indices + 1)
  author_map = np.loadtxt(os.path.join(data_dir, "author_map.txt"),
                          dtype=str, 
                          delimiter="\n")
  documents = random_state.permutation(num_documents)
  shuffled_author_indices = author_indices[documents]
  shuffled_counts = counts[documents]
  if counts_transformation == "nothing":
    count_values = shuffled_counts.data
  elif counts_transformation == "log":
    count_values = np.round(np.log(1 + shuffled_counts.data))
  else:
    raise ValueError("Unrecognized counts transformation.")
  shuffled_counts = tf.SparseTensor(
      indices=np.array(shuffled_counts.nonzero()).T, 
      values=count_values,
      dense_shape=shuffled_counts.shape)
  dataset = tf.data.Dataset.from_tensor_slices(
      ({"document_indices": documents, 
        "author_indices": shuffled_author_indices}, shuffled_counts))
  if train_iteratively:
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
  vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"), 
                          dtype=str, 
                          delimiter="\n",
                          comments="<!-")
  total_counts_per_author = np.bincount(
      author_indices, 
      weights=np.array(np.sum(counts, axis=1)).flatten())
  counts_per_document_per_author = (
      total_counts_per_author / np.bincount(author_indices))
  # Author weights is how lengthy each author's opinion over average is.
  author_weights = (counts_per_document_per_author / 
                    np.mean(np.sum(counts, axis=1))).astype(np.float32)
  return (dataset, author_weights, vocabulary, author_map,
          num_documents, num_words, num_authors)


class VariationalFamily(tf.keras.layers.Layer):
  def __init__(self, family, shape, initial_loc=None):
    super(VariationalFamily, self).__init__()
    self.family = family
    if initial_loc is None:
      self.location = tf.Variable(
          tf.keras.initializers.GlorotUniform()(shape=shape))
    else:
      self.location = tf.Variable(np.log(initial_loc))
   
    self.scale = tfp.util.TransformedVariable(
        tf.ones(shape) * 1.,
        bijector=tfp.bijectors.Softplus())
    if family == 'normal':
      self.distribution = tfp.distributions.Normal(loc=self.location,
                                                   scale=self.scale)
      self.prior = tfp.distributions.Normal(loc=0., scale=1.)
    elif family == 'lognormal':
      self.distribution = tfp.distributions.LogNormal(loc=self.location,
                                                      scale=self.scale)
      self.prior = tfp.distributions.Gamma(concentration=0.3, rate=0.3)
    else:
      raise ValueError("Unrecognized variational family.")
    
    # NOTE: This is a shortcoming of tf.keras.
    # See: https://github.com/tensorflow/probability/issues/946
    self.recognized_variables = self.distribution.variables  
    
  def get_log_prior(self, samples):
    # Sum all but first axis.
    log_prior = tf.reduce_sum(self.prior.log_prob(samples),
                              axis=tuple(range(1, len(samples.shape))))
    return log_prior
  
  def get_entropy(self, samples):
    # Sum all but first axis.
    entropy = -tf.reduce_sum(self.distribution.log_prob(samples),
                             axis=tuple(range(1, len(samples.shape))))
    return entropy
  
  def sample(self, num_samples):
    return self.distribution.sample(num_samples)


class TBIP(tf.keras.Model):
  def __init__(self, 
               training_iteratively,
               author_weights,
               initial_document_loc,
               initial_objective_topic_loc,
               num_samples):
    super(TBIP, self).__init__()
    self.training_iteratively = training_iteratively
    self.author_weights = author_weights
    num_documents, num_topics = initial_document_loc.shape
    _, num_words = initial_objective_topic_loc.shape
    num_authors = len(author_weights)
    self.document_distribution = VariationalFamily(
        'lognormal',
        [num_documents, num_topics],
        initial_loc=initial_document_loc)
    self.objective_topic_distribution = VariationalFamily(
        'lognormal',
        [num_topics, num_words],
        initial_loc=initial_objective_topic_loc)
    self.ideological_topic_distribution = VariationalFamily(
        'normal',
        [num_topics, num_words])
    self.ideal_point_distribution = VariationalFamily(
        'normal', 
        [num_authors])
    self.num_samples = num_samples
    
  def get_log_prior(self, 
                    document_samples, 
                    objective_topic_samples, 
                    ideological_topic_samples, 
                    ideal_point_samples):
    document_log_prior = self.document_distribution.get_log_prior(
        document_samples)
    objective_topic_log_prior = (
        self.objective_topic_distribution.get_log_prior(
            objective_topic_samples))
    ideological_topic_log_prior = (
        self.ideological_topic_distribution.get_log_prior(
            ideological_topic_samples))
    ideal_point_log_prior = self.ideal_point_distribution.get_log_prior(
        ideal_point_samples)
    log_prior = (document_log_prior + 
                 objective_topic_log_prior + 
                 ideological_topic_log_prior + 
                 ideal_point_log_prior)
    return log_prior
  
  def get_entropy(self, 
                  document_samples, 
                  objective_topic_samples, 
                  ideological_topic_samples, 
                  ideal_point_samples):
    document_entropy = self.document_distribution.get_entropy(
        document_samples)
    objective_topic_entropy = (
        self.objective_topic_distribution.get_entropy(objective_topic_samples))
    ideological_topic_entropy = (
        self.ideological_topic_distribution.get_entropy(
            ideological_topic_samples))
    ideal_point_entropy = self.ideal_point_distribution.get_entropy(
        ideal_point_samples)
    entropy = (document_entropy + 
               objective_topic_entropy + 
               ideological_topic_entropy + 
               ideal_point_entropy)
    return entropy
  
  def get_samples(self):
    document_samples = self.document_distribution.sample(
        self.num_samples)
    objective_topic_samples = self.objective_topic_distribution.sample(
        self.num_samples)
    ideological_topic_samples = self.ideological_topic_distribution.sample(
        self.num_samples)
    ideal_point_samples = self.ideal_point_distribution.sample(
        self.num_samples)
    samples = [document_samples, objective_topic_samples,
               ideological_topic_samples, ideal_point_samples]
    return samples
  
  def get_rate_log_prior_entropy(self, 
                                 document_indices=None, 
                                 author_indices=None):
    (document_samples, objective_topic_samples,
     ideological_topic_samples, ideal_point_samples) = self.get_samples()
    
    log_prior = self.get_log_prior(document_samples, 
                                   objective_topic_samples, 
                                   ideological_topic_samples, 
                                   ideal_point_samples)
    
    entropy = self.get_entropy(document_samples, 
                               objective_topic_samples, 
                               ideological_topic_samples, 
                               ideal_point_samples)
    rate = None
    if document_indices is not None and author_indices is not None:
      selected_document_samples = tf.gather(document_samples, 
                                            document_indices, 
                                            axis=1)
      selected_ideal_points = tf.gather(ideal_point_samples, 
                                        author_indices, 
                                        axis=1)
      selected_ideological_topic_samples = tf.exp(
          selected_ideal_points[:, :, tf.newaxis, tf.newaxis] *
          ideological_topic_samples[:, tf.newaxis, :, :])
      selected_author_weights = tf.gather(self.author_weights, author_indices)
      selected_ideological_topic_samples = (
          selected_author_weights[tf.newaxis, :, tf.newaxis, tf.newaxis] *
          selected_ideological_topic_samples) 
      rate = tf.reduce_sum(
          selected_document_samples[:, :, :, tf.newaxis] * 
          objective_topic_samples[:, tf.newaxis, :, :] * 
          selected_ideological_topic_samples[:, :, :, :],
          axis=2) 
    return rate, log_prior, entropy
    
  def get_topic_means(self):
    objective_topic_loc = self.objective_topic_distribution.location
    objective_topic_scale = self.objective_topic_distribution.scale
    ideological_topic_loc = self.ideological_topic_distribution.location
    ideological_topic_scale = self.ideological_topic_distribution.scale
    
    neutral_mean = objective_topic_loc + objective_topic_scale ** 2 / 2
    positive_mean = (objective_topic_loc + 
                     ideological_topic_loc + 
                     (objective_topic_scale ** 2 + 
                      ideological_topic_scale ** 2) / 2)
    negative_mean = (objective_topic_loc - 
                     ideological_topic_loc +
                     (objective_topic_scale ** 2 + 
                      ideological_topic_scale ** 2) / 2)
    return negative_mean, neutral_mean, positive_mean
  
  def call(self, inputs):
    document_indices = inputs['document_indices']
    author_indices = inputs['author_indices']
    # Scalars have extra dimensions when training with model.fit(). Weird.
    if not self.training_iteratively:
      document_indices = document_indices[:, 0]
      author_indices = author_indices[:, 0]

    rate, log_prior, entropy = self.get_rate_log_prior_entropy(
        document_indices, author_indices)
    self.add_loss(-tf.reduce_mean(log_prior))
    self.add_loss(-tf.reduce_mean(entropy))
    return rate


def create_loss_function(batch_size, num_documents):
  def get_negative_log_likelihood(counts, rate):
    # Take length of batch size because batch sizes can be variable (e.g. the
    # last batch of an epoch is shorter).
    batch_size = tf.shape(counts)[0]
    count_distribution = tfp.distributions.Poisson(rate=rate)
    count_log_likelihood = count_distribution.log_prob(
        tf.sparse.to_dense(counts))
    count_log_likelihood = tf.reduce_sum(count_log_likelihood, axis=[1, 2])
    # Adjust for the fact that we're only using a minibatch.
    count_log_likelihood = count_log_likelihood * float(num_documents / batch_size)
    return -count_log_likelihood
  return get_negative_log_likelihood


def print_topics(neutral_mean, negative_mean, positive_mean, vocabulary):
  """Get neutral and ideological topics to be used for Tensorboard.

  Args:
    neutral_mean: The mean of the neutral topics, a NumPy matrix with shape
      [num_topics, num_words].
    negative_mean: The mean of the negative topics, a NumPy matrix with shape
      [num_topics, num_words].
    positive_mean: The mean of the positive topics, a NumPy matrix with shape
      [num_topics, num_words].
    vocabulary: A list of the vocabulary with shape [num_words].

  Returns:
    topic_strings: A list of the negative, neutral, and positive topics.
  """
  num_topics, num_words = neutral_mean.shape
  words_per_topic = 10
  top_neutral_words = np.argsort(-neutral_mean, axis=1)
  top_negative_words = np.argsort(-negative_mean, axis=1)
  top_positive_words = np.argsort(-positive_mean, axis=1)
  topic_strings = []
  for topic_idx in range(num_topics):
    neutral_start_string = "Neutral {}:".format(topic_idx)
    neutral_row = [vocabulary[word] for word in 
                   top_neutral_words[topic_idx, :words_per_topic]]
    neutral_row_string = ", ".join(neutral_row)
    neutral_string = " ".join([neutral_start_string, neutral_row_string])
    
    positive_start_string = "Positive {}:".format(topic_idx)
    positive_row = [vocabulary[word] for word in 
                    top_positive_words[topic_idx, :words_per_topic]]
    positive_row_string = ", ".join(positive_row)
    positive_string = " ".join([positive_start_string, positive_row_string])
    
    negative_start_string = "Negative {}:".format(topic_idx)
    negative_row = [vocabulary[word] for word in 
                    top_negative_words[topic_idx, :words_per_topic]]
    negative_row_string = ", ".join(negative_row)
    negative_string = " ".join([negative_start_string, negative_row_string])
    
    topic_strings.append("  \n".join(
        [negative_string, neutral_string, positive_string]))
  return np.array(topic_strings)


def print_ideal_points(ideal_point_loc, author_map):
  """Print ideal point ordering for Tensorboard."""
  return ", ".join(author_map[np.argsort(ideal_point_loc)])


def log_static_features(tbip_model, vocabulary, author_map, step):
  negative_mean, neutral_mean, positive_mean = tbip_model.get_topic_means()
  ideal_point_list = print_ideal_points(
      tbip_model.ideal_point_distribution.location.numpy(),
      author_map)
  topics = print_topics(neutral_mean, 
                        negative_mean,
                        positive_mean,
                        vocabulary)
  tf.summary.text("ideal_points", ideal_point_list, step=step)
  tf.summary.text("topics", topics, step=step)
  tf.summary.histogram("params/document_loc", 
                       tbip_model.document_distribution.location, 
                       step=step)
  tf.summary.histogram("params/document_scale", 
                       tbip_model.document_distribution.scale, 
                       step=step)
  tf.summary.histogram("params/objective_topic_loc", 
                       tbip_model.objective_topic_distribution.location, 
                       step=step)
  tf.summary.histogram("params/objective_topic_scale", 
                       tbip_model.objective_topic_distribution.scale, 
                       step=step)
  tf.summary.histogram("params/ideological_topic_loc", 
                       tbip_model.ideological_topic_distribution.location, 
                       step=step)
  tf.summary.histogram("params/ideological_topic_scale", 
                       tbip_model.ideological_topic_distribution.scale, 
                       step=step)
  tf.summary.histogram("params/ideal_point_loc", 
                       tbip_model.ideal_point_distribution.location, 
                       step=step)
  tf.summary.histogram("params/ideal_point_scale", 
                       tbip_model.ideal_point_distribution.scale, 
                       step=step)


@tf.function
def train_step(tbip_model, inputs, outputs, loss_fn, optim):
  with tf.GradientTape() as tape:
    predictions = tbip_model(inputs)
    reconstruction_loss = loss_fn(outputs, predictions)
    # TODO(vafa): Name these losses.
    log_prior_loss, entropy_loss = tbip_model.losses
    total_loss = reconstruction_loss + log_prior_loss + entropy_loss
  trainable_variables = tape.watched_variables()
  grads = tape.gradient(total_loss, trainable_variables)
  optim.apply_gradients(zip(grads, trainable_variables))
  return total_loss, reconstruction_loss, log_prior_loss, entropy_loss


def main(argv):
  del argv
  tf.random.set_seed(FLAGS.seed)
  random_state = np.random.RandomState(FLAGS.seed)
  
  project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                             os.pardir)) 
  source_dir = os.path.join(project_dir, "data/{}".format(FLAGS.data))
  # For model comparisons, we must also specify a Senate session.
  if FLAGS.data == "senate-speech-comparisons":
    source_dir = os.path.join(
        source_dir, "tbip/{}".format(FLAGS.senate_session))
  # As described in the docstring, the data directory must have the following 
  # files: counts.npz, author_indices.npy, vocabulary.txt, author_map.txt.
  data_dir = os.path.join(source_dir, "clean")
  save_dir = os.path.join(source_dir, "tbip-tf2-fits")
  
  if tf.io.gfile.exists(save_dir):
    print("Deleting old log directory at {}".format(save_dir))
    tf.io.gfile.rmtree(save_dir)
  
  (dataset, author_weights, vocabulary, author_map, 
   num_documents, num_words, num_authors) = build_input_pipeline(
      data_dir, 
      FLAGS.batch_size,
      random_state,
      FLAGS.train_iteratively,
      FLAGS.counts_transformation)
   
  if FLAGS.pre_initialize_parameters:
    fit_dir = os.path.join(source_dir, "pf-fits")
    fitted_document_shape = np.load(
        os.path.join(fit_dir, "document_shape.npy")).astype(np.float32)
    fitted_document_rate = np.load(
        os.path.join(fit_dir, "document_rate.npy")).astype(np.float32)
    fitted_topic_shape = np.load(
        os.path.join(fit_dir, "topic_shape.npy")).astype(np.float32)
    fitted_topic_rate = np.load(
        os.path.join(fit_dir, "topic_rate.npy")).astype(np.float32)
    initial_document_loc = fitted_document_shape / fitted_document_rate
    initial_objective_topic_loc = fitted_topic_shape / fitted_topic_rate
  else:
    initial_document_loc = np.float32(
        np.exp(random_state.randn(num_documents, FLAGS.num_topics)))
    initial_objective_topic_loc = np.float32(
        np.exp(random_state.randn(FLAGS.num_topics, num_words)))
  
  def log_summaries(epoch, logs):
    with tensorboard_callback._get_writer('train').as_default():
      if epoch % 20 == 0:
        _, log_prior, entropy = tbip_model.get_rate_log_prior_entropy()
        tf.summary.scalar("elbo/entropy", 
                          tf.reduce_mean(entropy), 
                          step=epoch)
        tf.summary.scalar("elbo/log_prior", 
                          tf.reduce_mean(log_prior), 
                          step=epoch)
        
        log_static_features(tbip_model, vocabulary, author_map, epoch)
        
  optim = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate)
  loss_fn = create_loss_function(FLAGS.batch_size, num_documents)
  
  tbip_model = TBIP(FLAGS.train_iteratively,
                    author_weights, 
                    initial_document_loc, 
                    initial_objective_topic_loc, 
                    FLAGS.num_samples)
  if not FLAGS.train_iteratively:
    # Define the per-epoch callback.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir)
    # NOTE: We print loss after every epoch. There is no way to print less than
    # once per epoch using the tensorboard callback. We use the Tensorboard 
    # callback because that is the only way to log the reconstruction loss.
    summary_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=log_summaries)
    tbip_model.compile(optimizer=optim, loss=loss_fn)
    # NOTE: Fitting via compilation training prints summaries on the epoch 
    # level rather than on the batch level.
    num_epochs = int(FLAGS.max_steps * FLAGS.batch_size / num_documents)
    tbip_model.fit(dataset, 
                   epochs=num_epochs, 
                   callbacks=[tensorboard_callback, summary_callback])
  else:
    iterative_summary_writer = tf.summary.create_file_writer(save_dir)
    iterator = dataset.__iter__()
    start_time = time.time()
    for step in range(FLAGS.max_steps):
      inputs, outputs = iterator.next()
      (total_loss, reconstruction_loss, 
       log_prior_loss, entropy_loss) = train_step(
          tbip_model, inputs, outputs, loss_fn, optim)
      
      if step % FLAGS.print_steps == 0:
        duration = (time.time() - start_time) / (step + 1)
        print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec)".format(
            step, -total_loss.numpy()[0], duration))
        
        with iterative_summary_writer.as_default():
          tf.summary.scalar('loss', total_loss[0], step=step)
          tf.summary.scalar("elbo/entropy", -entropy_loss, step=step)
          tf.summary.scalar("elbo/log_prior", -log_prior_loss, step=step)
          tf.summary.scalar("elbo/count_log_likelihood", 
                            -reconstruction_loss[0], 
                            step=step)
          tf.summary.scalar('elbo/elbo', -total_loss[0], step=step)
          log_static_features(tbip_model, vocabulary, author_map, step)
        iterative_summary_writer.flush()
  
  
if __name__ == "__main__":
  app.run(main)
