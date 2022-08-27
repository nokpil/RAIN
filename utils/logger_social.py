import atexit
import json
import os
import os.path as osp
import shutil
import time
import warnings

import joblib
import numpy as np
import torch
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from user_config import (
    DEFAULT_DATA_DIR,
    DEFAULT_SHORTHAND,
    FORCE_DATESTAMP,
    WAIT_BEFORE_LAUNCH,
)

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    mean, std = x.mean(), x.std()

    if with_min_and_max:
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, global_min, global_max
    return mean, std


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, SummaryWriter):
            return str(obj)

        elif isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.
    If no seed is given and datestamp is false,
    ::
        output_dir = data_dir/exp_name
    If a seed is given and datestamp is false,
    ::
        output_dir = data_dir/exp_name/exp_name_s[seed]
    If datestamp is true, amend to
    ::
        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]
    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in
    ``user_config.py``.
    Args:
        exp_name (string): Name for experiment.
        seed (int): Seed for random number generators used by experiment.
        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``user_config.py``.
        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.
    Returns:
        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%y%b%d_") if datestamp else ""
    relpath = "".join([ymd_time, exp_name])

    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%y%b%d_%H-%M-%S")
            subfolder = "".join([hms_time, "-", exp_name, "_s", str(seed)])
        else:
            subfolder = "".join([exp_name, "_s", str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), exp_name=exp_name)
    return logger_kwargs


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


class Logger:
    """
    A general-purpose logger.
    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname="progress.txt", exp_name=None):
        """
        Initialize a Logger.
        Args:
            output_dir (string): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.
            output_fname (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress.txt``.
            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        print('logger started')
        self.output_dir = output_dir or "/tmp/experiments/%i" % int(time.time())
        if osp.exists(self.output_dir):
            print(
                "Warning: Log dir %s already exists! Storing info there anyway."
                % self.output_dir
            )
        else:
            os.makedirs(self.output_dir)
        self.output_file = open(osp.join(self.output_dir, output_fname), "w")
        atexit.register(self.output_file.close)
        print(
            colorize("Logging data to %s" % self.output_file.name, "green", bold=True)
        )
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
        self.writer = SummaryWriter(self.output_dir)

    def log(self, msg, color="green"):
        """Print a colorized message to stdout."""
        print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.
        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, (
                "Trying to introduce a new key %s that you didn't include in the first iteration"
                % key
            )
        assert key not in self.log_current_row, (
            "You already set %s this iteration. Maybe you forgot to call dump_tabular()"
            % key
        )
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.
        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).
        Example use:
        .. code-block:: python
            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json["exp_name"] = self.exp_name
        output = json.dumps(
            config_json, separators=(",", ":\t"), indent=4, sort_keys=True
        )
        print(colorize("Saving config:\n", color="cyan", bold=True))
        print(output)
        with open(osp.join(self.output_dir, "config.json"), "w") as out:
            out.write(output)

    def save_state(self, state_dict=None, itr=None):
        """
        Saves the state of an experiment.
        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you
        previously set up saving for with ``setup_tf_saver``.
        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.
        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.
            itr: An int, or None. Current iteration of training.
        """

        if hasattr(self, "pytorch_saver_elements"):
            self._pytorch_state_dict_saver(itr)
        else:
            fname = "vars.pkl" if itr is None else "vars%d.pkl" % itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except Exception:
                self.log("Warning: could not pickle state_dict.", color="red")

    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.
        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.
        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def _pytorch_state_dict_saver(self, itr=None):

        assert hasattr(
            self, "pytorch_saver_elements"
        ), "First have to setup saving with self.setup_pytorch_saver"
        fpath = "pyt_save"
        fpath = osp.join(self.output_dir, fpath)
        fname = "model" + ("%d" % itr if itr is not None else "") + ".pth"
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)
        save_dict = OrderedDict()

        for key in self.pytorch_saver_elements:
            save_dict[key] = self.pytorch_saver_elements[key].state_dict()
        torch.save(save_dict, fname)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.
        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False

    def close(self):
        self.writer.close()


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.
    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.
    With an EpochLogger, each time the quantity is calculated, you would
    use
    .. code-block:: python
        epoch_logger.store(NameOfQuantity=quantity_value)
    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use
    .. code-block:: python
        epoch_logger.log_tabular(NameOfQuantity, **options)
    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()
        self.baseline_data_dict = {'keys': []}

    def set_baseline(self, baseline_data_dict):
        self.baseline_data_dict = baseline_data_dict

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.
        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
            if key == "Epoch":
                self.epoch = val
        else:
            v = self.epoch_dict[key]
            vals = (
                np.concatenate(v)
                if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
                else v
            )
            stats = statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else "Avg" + key, stats[0])
            if key in self.baseline_data_dict['keys']:
                stat_dict = {}
                stat_dict['RL'] = stats[0]
                
                for baseline_name in self.baseline_data_dict:
                    if baseline_name != 'keys':
                        stat_dict[baseline_name] = self.baseline_data_dict[baseline_name][key]
                self.writer.add_scalars(
                    key if average_only else key + "/Avg", stat_dict, self.epoch
                )
            else:
                self.writer.add_scalar(
                    key if average_only else key + "/Avg", stats[0], self.epoch
                )
            if not (average_only):
                super().log_tabular("Std" + key, stats[1])
                self.writer.add_scalar(key + "/Std", stats[1], self.epoch)
            if with_min_and_max:
                super().log_tabular("Max" + key, stats[3])
                super().log_tabular("Min" + key, stats[2])
                self.writer.add_scalar(key + "/Max", stats[3], self.epoch)
                self.writer.add_scalar(key + "/Min", stats[2], self.epoch)
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = (
            np.concatenate(v)
            if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
            else v
        )
        return statistics_scalar(vals)
