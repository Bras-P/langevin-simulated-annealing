import tensorflow as tf

class InverseRLogDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
#
  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      decay_rate,
      staircase=False,
      name=None):
    super(InverseRLogDecay, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.staircase = staircase
    self.name = name
#
  def __call__(self, step):
    with tf.name_scope(self.name or "InverseRLogDecay") as name:
      initial_learning_rate = tf.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = tf.cast(self.decay_steps, dtype)
      decay_rate = tf.cast(self.decay_rate, dtype)
      global_step_recomp = tf.cast(step, dtype)
      p = global_step_recomp / decay_steps
      if self.staircase:
        p = tf.floor(p)
      const = tf.cast(tf.math.exp(1.)*tf.constant(1.), dtype)
      denom = tf.add(const, tf.multiply(decay_rate, p))
      return initial_learning_rate * tf.math.rsqrt(tf.math.log(denom))
#
  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "decay_rate": self.decay_rate,
        "staircase": self.staircase,
        "name": self.name
    }
