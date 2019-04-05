# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bring in all of the public TensorFlow interface into this module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import distutils as _distutils
import inspect as _inspect
import os as _os
import site as _site
import sys as _sys

from tensorflow._api.v2 import audio
from tensorflow._api.v2 import autograph
from tensorflow._api.v2 import bitwise
from tensorflow._api.v2 import compat
from tensorflow._api.v2 import config
from tensorflow._api.v2 import data
from tensorflow._api.v2 import debugging
from tensorflow._api.v2 import distribute
from tensorflow._api.v2 import dtypes
from tensorflow._api.v2 import errors
from tensorflow._api.v2 import experimental
from tensorflow._api.v2 import feature_column
from tensorflow._api.v2 import graph_util
from tensorflow._api.v2 import image
from tensorflow._api.v2 import io
from tensorflow._api.v2 import linalg
from tensorflow._api.v2 import lite
from tensorflow._api.v2 import lookup
from tensorflow._api.v2 import math
from tensorflow._api.v2 import nest
from tensorflow._api.v2 import nn
from tensorflow._api.v2 import quantization
from tensorflow._api.v2 import queue
from tensorflow._api.v2 import ragged
from tensorflow._api.v2 import random
from tensorflow._api.v2 import raw_ops
from tensorflow._api.v2 import saved_model
from tensorflow._api.v2 import sets
from tensorflow._api.v2 import signal
from tensorflow._api.v2 import sparse
from tensorflow._api.v2 import strings
from tensorflow._api.v2 import summary
from tensorflow._api.v2 import sysconfig
from tensorflow._api.v2 import test
from tensorflow._api.v2 import tpu
from tensorflow._api.v2 import train
from tensorflow._api.v2 import version
from tensorflow.lite.python.lite import _import_graph_def as import_graph_def
from tensorflow.python import AggregationMethod
from tensorflow.python import Assert
from tensorflow.python import CriticalSection
from tensorflow.python import DType
from tensorflow.python import GradientTape
from tensorflow.python import Graph
from tensorflow.python import IndexedSlices
from tensorflow.python import NoGradient as no_gradient
from tensorflow.python import Operation
from tensorflow.python import RegisterGradient
from tensorflow.python import SparseTensor
from tensorflow.python import Tensor
from tensorflow.python import TensorArray
from tensorflow.python import TensorShape
from tensorflow.python import UnconnectedGradients
from tensorflow.python import Variable
from tensorflow.python import VariableAggregationV2 as VariableAggregation
from tensorflow.python import VariableSynchronization
from tensorflow.python import abs
from tensorflow.python import acos
from tensorflow.python import acosh
from tensorflow.python import add
from tensorflow.python import add_n
from tensorflow.python import argmax_v2 as argmax
from tensorflow.python import argmin_v2 as argmin
from tensorflow.python import argsort
from tensorflow.python import as_dtype
from tensorflow.python import as_string
from tensorflow.python import asin
from tensorflow.python import asinh
from tensorflow.python import atan
from tensorflow.python import atan2
from tensorflow.python import atanh
from tensorflow.python import batch_to_space_v2 as batch_to_space
from tensorflow.python import bitcast
from tensorflow.python import boolean_mask_v2 as boolean_mask
from tensorflow.python import broadcast_dynamic_shape
from tensorflow.python import broadcast_static_shape
from tensorflow.python import broadcast_to
from tensorflow.python import case
from tensorflow.python import cast
from tensorflow.python import clip_by_global_norm
from tensorflow.python import clip_by_norm
from tensorflow.python import clip_by_value
from tensorflow.python import complex
from tensorflow.python import concat
from tensorflow.python import constant
from tensorflow.python import control_dependencies
from tensorflow.python import cos
from tensorflow.python import cosh
from tensorflow.python import cumsum
from tensorflow.python import custom_gradient
from tensorflow.python import divide
from tensorflow.python import dynamic_partition
from tensorflow.python import dynamic_stitch
from tensorflow.python import edit_distance
from tensorflow.python import einsum
from tensorflow.python import equal
from tensorflow.python import executing_eagerly
from tensorflow.python import exp
from tensorflow.python import expand_dims_v2 as expand_dims
from tensorflow.python import extract_volume_patches
from tensorflow.python import eye
from tensorflow.python import fill
from tensorflow.python import floor
from tensorflow.python import floor_div
from tensorflow.python import floor_mod as floormod
from tensorflow.python import floor_mod as mod
from tensorflow.python import foldl
from tensorflow.python import foldr
from tensorflow.python import function
from tensorflow.python import gather_nd
from tensorflow.python import gather_v2 as gather
from tensorflow.python import greater
from tensorflow.python import greater_equal
from tensorflow.python import group
from tensorflow.python import guarantee_const
from tensorflow.python import histogram_fixed_width
from tensorflow.python import histogram_fixed_width_bins
from tensorflow.python import identity
from tensorflow.python import identity_n
from tensorflow.python import less
from tensorflow.python import less_equal
from tensorflow.python import lin_space as linspace
from tensorflow.python import load_library
from tensorflow.python import load_op_library
from tensorflow.python import logical_and
from tensorflow.python import logical_not
from tensorflow.python import logical_or
from tensorflow.python import make_ndarray
from tensorflow.python import matmul
from tensorflow.python import matrix_square_root
from tensorflow.python import maximum
from tensorflow.python import meshgrid
from tensorflow.python import minimum
from tensorflow.python import multiply
from tensorflow.python import neg as negative
from tensorflow.python import no_op
from tensorflow.python import no_regularizer
from tensorflow.python import norm_v2 as norm
from tensorflow.python import not_equal
from tensorflow.python import one_hot
from tensorflow.python import ones
from tensorflow.python import ones_like_v2 as ones_like
from tensorflow.python import pad_v2 as pad
from tensorflow.python import parallel_stack
from tensorflow.python import pow
from tensorflow.python import range
from tensorflow.python import rank
from tensorflow.python import real_div as realdiv
from tensorflow.python import reduce_all
from tensorflow.python import reduce_any
from tensorflow.python import reduce_logsumexp
from tensorflow.python import reduce_max
from tensorflow.python import reduce_mean
from tensorflow.python import reduce_min
from tensorflow.python import reduce_prod
from tensorflow.python import reduce_sum
from tensorflow.python import register_tensor_conversion_function
from tensorflow.python import required_space_to_batch_paddings
from tensorflow.python import reshape
from tensorflow.python import reverse
from tensorflow.python import reverse_sequence_v2 as reverse_sequence
from tensorflow.python import roll
from tensorflow.python import round
from tensorflow.python import saturate_cast
from tensorflow.python import scalar_mul_v2 as scalar_mul
from tensorflow.python import scan
from tensorflow.python import scatter_nd
from tensorflow.python import searchsorted
from tensorflow.python import sequence_mask
from tensorflow.python import shape_n
from tensorflow.python import shape_v2 as shape
from tensorflow.python import sigmoid
from tensorflow.python import sign
from tensorflow.python import sin
from tensorflow.python import sinh
from tensorflow.python import size_v2 as size
from tensorflow.python import slice
from tensorflow.python import sort
from tensorflow.python import space_to_batch_nd
from tensorflow.python import space_to_batch_v2 as space_to_batch
from tensorflow.python import split
from tensorflow.python import sqrt
from tensorflow.python import square
from tensorflow.python import squeeze_v2 as squeeze
from tensorflow.python import stack
from tensorflow.python import stop_gradient
from tensorflow.python import strided_slice
from tensorflow.python import string_split
from tensorflow.python import subtract
from tensorflow.python import tan
from tensorflow.python import tanh
from tensorflow.python import tensor_scatter_add as tensor_scatter_nd_add
from tensorflow.python import tensor_scatter_sub as tensor_scatter_nd_sub
from tensorflow.python import tensor_scatter_update as tensor_scatter_nd_update
from tensorflow.python import tensordot
from tensorflow.python import tile
from tensorflow.python import timestamp
from tensorflow.python import transpose_v2 as transpose
from tensorflow.python import truediv
from tensorflow.python import truncate_div as truncatediv
from tensorflow.python import truncate_mod as truncatemod
from tensorflow.python import unique
from tensorflow.python import unique_with_counts
from tensorflow.python import unravel_index
from tensorflow.python import unstack
from tensorflow.python import where
from tensorflow.python import zeros
from tensorflow.python import zeros_like_v2 as zeros_like
from tensorflow.python.framework.dtypes import bfloat16
from tensorflow.python.framework.dtypes import bool
from tensorflow.python.framework.dtypes import complex128
from tensorflow.python.framework.dtypes import complex64
from tensorflow.python.framework.dtypes import double
from tensorflow.python.framework.dtypes import float16
from tensorflow.python.framework.dtypes import float32
from tensorflow.python.framework.dtypes import float64
from tensorflow.python.framework.dtypes import half
from tensorflow.python.framework.dtypes import int16
from tensorflow.python.framework.dtypes import int32
from tensorflow.python.framework.dtypes import int64
from tensorflow.python.framework.dtypes import int8
from tensorflow.python.framework.dtypes import qint16
from tensorflow.python.framework.dtypes import qint32
from tensorflow.python.framework.dtypes import qint8
from tensorflow.python.framework.dtypes import quint16
from tensorflow.python.framework.dtypes import quint8
from tensorflow.python.framework.dtypes import resource
from tensorflow.python.framework.dtypes import string
from tensorflow.python.framework.dtypes import uint16
from tensorflow.python.framework.dtypes import uint32
from tensorflow.python.framework.dtypes import uint64
from tensorflow.python.framework.dtypes import uint8
from tensorflow.python.framework.dtypes import variant
from tensorflow.python.framework.ops import convert_to_tensor_v2 as convert_to_tensor
from tensorflow.python.framework.ops import device_v2 as device
from tensorflow.python.framework.ops import init_scope
from tensorflow.python.framework.ops import name_scope_v2 as name_scope
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.framework.tensor_util import constant_value as get_static_value
from tensorflow.python.framework.tensor_util import is_tensor
from tensorflow.python.framework.versions import COMPILER_VERSION as __compiler_version__
from tensorflow.python.framework.versions import CXX11_ABI_FLAG as __cxx11_abi_flag__
from tensorflow.python.framework.versions import GIT_VERSION as __git_version__
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as __monolithic_build__
from tensorflow.python.framework.versions import VERSION as __version__
from tensorflow.python.keras.initializers import ConstantV2 as constant_initializer
from tensorflow.python.keras.initializers import OnesV2 as ones_initializer
from tensorflow.python.keras.initializers import RandomNormalV2 as random_normal_initializer
from tensorflow.python.keras.initializers import RandomUniformV2 as random_uniform_initializer
from tensorflow.python.keras.initializers import ZerosV2 as zeros_initializer
from tensorflow.python.module.module import Module
from tensorflow.python.ops.array_ops import newaxis
from tensorflow.python.ops.check_ops import assert_equal_v2 as assert_equal
from tensorflow.python.ops.check_ops import assert_greater_v2 as assert_greater
from tensorflow.python.ops.check_ops import assert_less_v2 as assert_less
from tensorflow.python.ops.check_ops import assert_rank_v2 as assert_rank
from tensorflow.python.ops.check_ops import ensure_shape
from tensorflow.python.ops.control_flow_grad import cond_for_tf_v2 as cond
from tensorflow.python.ops.control_flow_grad import tuple_v2 as tuple
from tensorflow.python.ops.control_flow_grad import while_loop_v2 as while_loop
from tensorflow.python.ops.gen_image_ops import combined_non_max_suppression
from tensorflow.python.ops.gradients_impl import HessiansV2 as hessians
from tensorflow.python.ops.gradients_impl import gradients_v2 as gradients
from tensorflow.python.ops.logging_ops import print_v2 as print
from tensorflow.python.ops.map_fn import map_fn
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.script_ops import eager_py_func as py_function
from tensorflow.python.ops.script_ops import numpy_function
from tensorflow.python.ops.variable_scope import variable_creator_scope
from tensorflow.python.platform.tf_logging import get_logger
_names_with_underscore = ['__version__', '__git_version__', '__compiler_version__', '__cxx11_abi_flag__', '__monolithic_build__']
__all__ = [_s for _s in dir() if not _s.startswith('_')]
__all__.extend([_s for _s in _names_with_underscore])


# Make sure directory containing top level submodules is in
# the __path__ so that "from tensorflow.foo import bar" works.
# We're using bitwise, but there's nothing special about that.
_API_MODULE = bitwise  # pylint: disable=undefined-variable
_current_module = _sys.modules[__name__]
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
if not hasattr(_current_module, '__path__'):
  __path__ = [_tf_api_dir]
elif _tf_api_dir not in __path__:
  __path__.append(_tf_api_dir)

# pylint: disable=g-bad-import-order
from tensorflow.python.tools import component_api_helper as _component_api_helper
_component_api_helper.package_hook(
    parent_package_str=__name__,
    child_package_str=('tensorboard.summary._tf.summary'),
    error_msg="Limited tf.summary API due to missing TensorBoard installation")
_component_api_helper.package_hook(
    parent_package_str=__name__,
    child_package_str=(
        'tensorflow_estimator.python.estimator.api._v2.estimator'))

if not hasattr(_current_module, 'estimator'):
  _component_api_helper.package_hook(
      parent_package_str=__name__,
      child_package_str=(
          'tensorflow_estimator.python.estimator.api.estimator'))
_component_api_helper.package_hook(
    parent_package_str=__name__,
    child_package_str=('tensorflow.python.keras.api._v2.keras'))

# Enable TF2 behaviors
from tensorflow.python.compat import v2_compat as _compat  # pylint: disable=g-import-not-at-top
_compat.enable_v2_behavior()


# Load all plugin libraries from site-packages/tensorflow-plugins if we are
# running under pip.
# TODO(gunan): Enable setting an environment variable to define arbitrary plugin
# directories.
# TODO(gunan): Find a better location for this code snippet.
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi

# Get sitepackages directories for the python installation.
_site_packages_dirs = []
_site_packages_dirs += [_site.USER_SITE]
_site_packages_dirs += [_p for _p in _sys.path if 'site-packages' in _p]
if 'getsitepackages' in dir(_site):
  _site_packages_dirs += _site.getsitepackages()

if 'sysconfig' in dir(_distutils):
  _site_packages_dirs += [_distutils.sysconfig.get_python_lib()]

_site_packages_dirs = list(set(_site_packages_dirs))

# Find the location of this exact file.
_current_file_location = _inspect.getfile(_inspect.currentframe())

def _running_from_pip_package():
  return any(
      _current_file_location.startswith(dir_) for dir_ in _site_packages_dirs)

if _running_from_pip_package():
  for s in _site_packages_dirs:
    # TODO(gunan): Add sanity checks to loaded modules here.
    plugin_dir = _os.path.join(s, 'tensorflow-plugins')
    if _fi.file_exists(plugin_dir):
      _ll.load_library(plugin_dir)

# These symbols appear because we import the python package which
# in turn imports from tensorflow.core and tensorflow.python. They
# must come from this module. So python adds these symbols for the
# resolution to succeed.
# pylint: disable=undefined-variable
try:
  del python
  del core
except NameError:
  # Don't fail if these modules are not available.
  # For e.g. this file will be originally placed under tensorflow/_api/v1 which
  # does not have 'python', 'core' directories. Then, it will be copied
  # to tensorflow/ which does have these two directories.
  pass
# Similarly for compiler. Do it separately to make sure we do this even if the
# others don't exist.
try:
  del compiler
except NameError:
  pass

# Add module aliases
if hasattr(_current_module, 'keras'):
  losses = keras.losses
  metrics = keras.metrics
  optimizers = keras.optimizers
  initializers = keras.initializers

# pylint: enable=undefined-variable
