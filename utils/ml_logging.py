import os
import tensorflow as tf
import numpy as np

# Base directory for logs
base_log_dir = "./logs"

def get_log_dir(expt_name):
    """
    Get the log directory for a given experiment name.

    Args:
        expt_name (str): The name of the experiment.

    Returns:
        str: Path to the log directory.
    """
    return os.path.join(base_log_dir, expt_name)

class Logger:
    """
    Logger class for logging scalar values and histograms during training.
    """
    def __init__(self, log_dir):
        """
        Initialize the Logger with the specified log directory.

        Args:
            log_dir (str): Directory to store log files.
        """
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """
        Log a scalar value.

        Args:
            tag (str): Name of the scalar.
            value (float): Value to log.
            step (int): Training step.
        """
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """
        Log a histogram of values.

        Args:
            tag (str): Name of the histogram.
            values (numpy.ndarray): Array of values.
            step (int): Training step.
            bins (int): Number of histogram bins.
        """
        counts, bin_edges = np.histogram(values, bins=bins)
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()

    def close(self):
        """Close the logger."""
        self.writer.close()
