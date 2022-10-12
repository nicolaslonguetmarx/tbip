"""The text-based ideal point model (TBIP) using Tensorflow 2.0."""
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
flags.DEFINE_integer("num_epochs",
                     default=10000,
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
flags.DEFINE_string("data",
                    default="senate-speeches-114",
                    help="Data source being used.")
flags.DEFINE_integer("save_every",
                     default=20,
                     help="Number of epochs after which to save and log")
flags.DEFINE_integer("seed",
                     default=123,
                     help="Random seed to be used.")
flags.DEFINE_string("checkpoint_name",
                    default="tmp",
                    help="Name to be used for saving results.")
flags.DEFINE_boolean("load_checkpoint",
                     default=True,
                     help="Whether to load checkpoint (only if it exists).")
FLAGS = flags.FLAGS


def build_input_pipeline(data_dir,
                         batch_size,
                         random_state,
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
  author_map = np.loadtxt(os.path.join(data_dir, "author_map.txt"),
                          dtype=str,
                          delimiter="\n")
  #nlm: potentially add some encoding here (e.g., 'latin-1')
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
  dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(
    batch_size)
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
          num_documents, num_words)


class VariationalFamily(tf.keras.layers.Layer):
  def __init__(self, family, shape, initial_loc=None):
    super(VariationalFamily, self).__init__()
    if initial_loc is None:
      self.location = tf.Variable(
        tf.keras.initializers.GlorotUniform()(shape=shape))
    else:
      self.location = tf.Variable(np.log(initial_loc))
    
    self.scale = tfp.util.TransformedVariable(
      tf.ones(shape),
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
  
  def sample(self, num_samples, seed=None):
    seed, sample_seed = tfp.random.split_seed(seed)
    return self.distribution.sample(num_samples, seed=sample_seed), seed


class TBIP(tf.keras.Model):
  def __init__(self,
               author_weights,
               initial_document_loc,
               initial_objective_topic_loc,
               batch_size,
               num_samples,):
    super(TBIP, self).__init__()
    self.author_weights = author_weights
    num_documents, num_topics = initial_document_loc.shape
    _, num_words = initial_objective_topic_loc.shape
    num_authors = len(author_weights)
    self.num_documents = num_documents
    self.batch_size = batch_size
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
    #nlm: here changed for num_authors, num_topics
    self.ideal_point_distribution = VariationalFamily(
      'normal',
      [num_authors, num_topics])
    self.num_samples = num_samples
  
  def get_log_prior(self,
                    document_samples,
                    objective_topic_samples,
                    ideological_topic_samples,
                    ideal_point_samples):
    document_log_prior = self.document_distribution.get_log_prior(
      document_samples)
    objective_topic_log_prior = (
      self.objective_topic_distribution.get_log_prior(objective_topic_samples))
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
  
  def get_samples(self, seed=None):
    document_samples, seed = self.document_distribution.sample(
      self.num_samples, seed=seed)
    objective_topic_samples, seed = self.objective_topic_distribution.sample(
      self.num_samples, seed=seed)
    ideological_topic_samples, seed = (
      self.ideological_topic_distribution.sample(self.num_samples, seed=seed))
    ideal_point_samples, seed = self.ideal_point_distribution.sample(
        self.num_samples, seed=seed)
    samples = [document_samples, objective_topic_samples,
               ideological_topic_samples, ideal_point_samples]
    return samples, seed
  
  def get_rate_log_prior_entropy(self, 
                                 document_indices, 
                                 author_indices, 
                                 seed=None):
    ((document_samples, objective_topic_samples,
     ideological_topic_samples, ideal_point_samples),
     seed) = self.get_samples(seed)
    log_prior = self.get_log_prior(document_samples,
                                   objective_topic_samples,
                                   ideological_topic_samples,
                                   ideal_point_samples)
    entropy = self.get_entropy(document_samples,
                               objective_topic_samples,
                               ideological_topic_samples,
                               ideal_point_samples)
    selected_document_samples = tf.gather(document_samples,
                                          document_indices,
                                          axis=1)
    selected_ideal_points = tf.gather(ideal_point_samples,
                                      author_indices,
                                      axis=1)
    #nlm: here replace by a column 
    selected_ideological_topic_samples = tf.exp(
      selected_ideal_points[:, :, :, tf.newaxis] *
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
    return rate, log_prior, entropy, seed
  
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
  
  def call(self, inputs, outputs, seed):
    document_indices = inputs['document_indices']
    author_indices = inputs['author_indices']
    rate, log_prior, entropy, seed = self.get_rate_log_prior_entropy(
      document_indices, author_indices, seed)
    return rate, -tf.reduce_mean(log_prior), -tf.reduce_mean(entropy), seed


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


def log_static_features(model, vocabulary, author_map, step):
  negative_mean, neutral_mean, positive_mean = model.get_topic_means()
  ideal_point_list = print_ideal_points(
    model.ideal_point_distribution.location.numpy(),
    author_map)
  topics = print_topics(neutral_mean,
                        negative_mean,
                        positive_mean,
                        vocabulary)
  tf.summary.text("ideal_points", ideal_point_list, step=step)
  tf.summary.text("topics", topics, step=step)

  tf.summary.histogram("params/document_loc",
                       model.document_distribution.location,
                       step=step)
  tf.summary.histogram("params/document_scale",
                       model.document_distribution.scale,
                       step=step)
  tf.summary.histogram("params/objective_topic_loc",
                       model.objective_topic_distribution.location,
                       step=step)
  tf.summary.histogram("params/objective_topic_scale",
                       model.objective_topic_distribution.scale,
                       step=step)
  tf.summary.histogram("params/ideological_topic_loc",
                       model.ideological_topic_distribution.location,
                       step=step)
  tf.summary.histogram("params/ideological_topic_scale",
                       model.ideological_topic_distribution.scale,
                       step=step)
  tf.summary.histogram("params/ideal_point_loc",
                       model.ideal_point_distribution.location,
                       step=step)
  tf.summary.histogram("params/ideal_point_scale",
                       model.ideal_point_distribution.scale,
                       step=step)


@tf.function
def train_step(model, inputs, outputs, optim, seed):
  with tf.GradientTape() as tape:
    predictions, log_prior_loss, entropy_loss, seed = model(
      inputs, outputs, seed)
    count_distribution = tfp.distributions.Poisson(rate=predictions)
    count_log_likelihood = count_distribution.log_prob(
      tf.sparse.to_dense(outputs))
    count_log_likelihood = tf.reduce_sum(count_log_likelihood, axis=[1, 2])
    # Adjust for the fact that we're only using a minibatch.
    batch_size = tf.shape(outputs)[0]
    count_log_likelihood = count_log_likelihood * tf.dtypes.cast(
      model.num_documents / batch_size, tf.float32)
    reconstruction_loss = -count_log_likelihood
    total_loss = tf.reduce_mean(reconstruction_loss + 
                                log_prior_loss + 
                                entropy_loss)
  trainable_variables = tape.watched_variables()
  grads = tape.gradient(total_loss, trainable_variables)
  optim.apply_gradients(zip(grads, trainable_variables))
  return total_loss, reconstruction_loss, log_prior_loss, entropy_loss, seed


def main(argv):
  del argv
  # Initial random seed for parameter initialization.
  tf.random.set_seed(FLAGS.seed)
  random_state = np.random.RandomState(FLAGS.seed)
  project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir))
  source_dir = os.path.join(project_dir, "data/{}".format(FLAGS.data))
  
  # As described in the docstring, the data directory must have the following
  # files: counts.npz, author_indices.npy, vocabulary.txt, author_map.txt.
  data_dir = os.path.join(source_dir, "clean")
  save_dir = os.path.join(source_dir, "fits/{}".format(FLAGS.checkpoint_name))

  (dataset, author_weights, vocabulary, author_map,
   num_documents, num_words) = build_input_pipeline(
      data_dir,
      FLAGS.batch_size,
      random_state,
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
  
  optim = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate)
  
  model = TBIP(author_weights,
               initial_document_loc,
               initial_objective_topic_loc,
               FLAGS.batch_size,
               FLAGS.num_samples,)
  # Add start epoch so checkpoint state is saved.
  model.start_epoch = tf.Variable(-1)

  checkpoint_dir = os.path.join(save_dir, "checkpoints")
  if os.path.exists(checkpoint_dir) and FLAGS.load_checkpoint:
    pass
  else:
    # If we're not loading a checkpoint, overwrite the existing directory
    # with saved results.
    if os.path.exists(save_dir):
      print("Deleting old log directory at {}".format(save_dir))
      tf.io.gfile.rmtree(save_dir)
  
  # We keep track of the seed to make sure the random number state is the same
  # whether or not we load a model.
  _, seed = tfp.random.split_seed(FLAGS.seed)
  checkpoint = tf.train.Checkpoint(optimizer=optim, 
                                   net=model, 
                                   seed=tf.Variable(seed))
  manager = tf.train.CheckpointManager(checkpoint, 
                                       checkpoint_dir, 
                                       max_to_keep=1)

  checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    # Load from saved checkpoint, keeping track of the seed.
    seed = checkpoint.seed
    # Since the dataset shuffles at every epoch and we'd like the runs to be
    # identical whether or not we load a checkpoint, we need to make sure the
    # dataset state is consistent. This is a hack but it will do for now.
    # Here's the issue: https://github.com/tensorflow/tensorflow/issues/48178
    for _ in range(model.start_epoch.numpy() + 1):
      _ = iter(dataset)
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  summary_writer = tf.summary.create_file_writer(save_dir)
  summary_writer.set_as_default()
  start_time = time.time()
  start_epoch = model.start_epoch.numpy()
  for epoch in range(start_epoch + 1, FLAGS.num_epochs):
    for batch_index, batch in enumerate(iter(dataset)):
      batches_per_epoch = len(dataset)
      step = batches_per_epoch * epoch + batch_index
      inputs, outputs = batch
      (total_loss, reconstruction_loss, 
       log_prior_loss, entropy_loss, seed) = train_step(
          model, inputs, outputs, optim, seed)
      checkpoint.seed.assign(seed)

    duration = (time.time() - start_time) / (epoch - start_epoch)
    print("Epoch: {:>3d} ELBO: {:.3f} Entropy: {:.1f} ({:.3f} sec)".format(
      epoch, -total_loss.numpy(), -entropy_loss.numpy(), duration))

    # Log to tensorboard at the end of every `save_every` epochs.
    if epoch % FLAGS.save_every == 0:
      tf.summary.scalar('loss', total_loss, step=step)
      tf.summary.scalar("elbo/entropy", -entropy_loss, step=step)
      tf.summary.scalar("elbo/log_prior", -log_prior_loss, step=step)
      tf.summary.scalar("elbo/count_log_likelihood", 
                        -tf.reduce_mean(reconstruction_loss), 
                        step=step)
      tf.summary.scalar('elbo/elbo', -total_loss, step=step)
      log_static_features(model, vocabulary, author_map, step)
      summary_writer.flush()

      # Save checkpoint too.
      model.start_epoch.assign(epoch)
      save_path = manager.save()
      print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

  
if __name__ == "__main__":
  app.run(main)
