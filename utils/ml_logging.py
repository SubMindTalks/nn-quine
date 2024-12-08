import os
import tensorflow as tf
import numpy as np

base_log_dir = "./logs"

def get_log_dir(expt_name):
    return os.path.join(base_log_dir, expt_name)

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Create and write Summary
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()

    def close(self):
        self.writer.close()
