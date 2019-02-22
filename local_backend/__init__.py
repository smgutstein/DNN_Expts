from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
import importlib


# Set Keras base dir path given KERAS_HOME env variable, if applicable.
# Otherwise either ~/.keras or /tmp.
if 'KERAS_HOME' in os.environ:
    _keras_dir = os.environ.get('KERAS_HOME')
else:
    _keras_base_dir = os.path.expanduser('~')
    if not os.access(_keras_base_dir, os.W_OK):
        _keras_base_dir = '/tmp'
    _keras_dir = os.path.join(_keras_base_dir, '.keras')

# Default backend: TensorFlow.
_LOCAL_BACKEND = 'tensorflow'

# Attempt to read Keras config file.
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _backend = _config.get('backend', _LOCAL_BACKEND)
    _LOCAL_BACKEND = _backend


# Set backend based on KERAS_BACKEND flag, if applicable.
if 'KERAS_BACKEND' in os.environ:
    _backend = os.environ['KERAS_BACKEND']
    if _backend:
        _LOCAL_BACKEND = _backend

# Import backend functions.
if _LOCAL_BACKEND == 'cntk':
    sys.stderr.write('Using CNTK local_backend\n')
    from .local_cntk_backend import *
elif _LOCAL_BACKEND == 'theano':
    sys.stderr.write('Using Theano local_backend.\n')
    from .local_theano_backend import *
elif _LOCAL_BACKEND == 'tensorflow':
    sys.stderr.write('Using TensorFlow local_backend.\n')
    from .local_tensorflow_backend import *
else:
    # Try and load external backend.
    try:
        backend_module = importlib.import_module(_LOCAL_BACKEND)
        entries = backend_module.__dict__
        # Check if valid backend.
        # Module is a valid backend if it has the required entries.
        required_entries = ['placeholder', 'variable', 'function']
        for e in required_entries:
            if e not in entries:
                raise ValueError('Invalid backend. Missing required entry : ' + e)
        namespace = globals()
        for k, v in entries.items():
            # Make sure we don't override any entries from common, such as epsilon.
            if k not in namespace:
                namespace[k] = v
        sys.stderr.write('Using ' + _LOCAL_BACKEND + ' backend.\n')
    except ImportError:
        raise ValueError('Unable to import backend : ' + str(_LOCAL_BACKEND))


def local_backend():
    """Publicly accessible method
    for determining the current backend.

    # Returns
        String, the name of the backend Keras is currently using.

    # Example
    ```python
        >>> keras.backend.backend()
        'tensorflow'
    ```
    """
    return _LOCAL_BACKEND
