# This file is derived from work originally created by Simon Hofmann et al.
# Original project: FaceSim3D (https://github.com/SHEscher/FaceSim3D)
#
# Copyright (c) 2023 Simon M. Hofmann et al. (MPI CBS)
#
# Licensed under the MIT License.
# See the LICENSE file in the project root or
# https://opensource.org/licenses/MIT

"""`facesim3d`: A Python package to reproduce the `FaceSim3D` study."""

__author__ = "Simon M. Hofmann"
__version__ = "1.0.1"
__year__ = "2022-2024"

import warnings

import facesim3d.configs

warnings.filterwarnings(action="ignore", category=FutureWarning)
