# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from se3_transformer.runtime.callbacks import BaseCallback
from se3_transformer.runtime.loggers import Logger
from se3_transformer.runtime.metrics import MeanAbsoluteError
from trip.runtime.metrics import RootMeanSquaredError

from se3_transformer.runtime.callbacks import LRSchedulerCallback


class TrIPMetricCallback(BaseCallback):
    """ Logs the recaled MAE and RMSE for TrIP regression """

    def __init__(self, logger, targets_std, prefix=''):
        self.mae = MeanAbsoluteError()
        self.rmse = RootMeanSquaredError()
        self.logger = logger
        self.targets_std = targets_std
        self.prefix = prefix
        self.best_mae = float('inf')
        self.best_rmse = float('inf')

    def on_validation_step(self, inputs, targets, preds):
        if 'energy' in self.prefix:
            pred = preds[0]
            target = targets['energy']
        elif 'forces' in self.prefix:
            pred = preds[1]
            target = targets['forces']
        self.rmse(pred.detach(), target.detach())
        self.mae(pred.detach(), target.detach())

    def on_validation_end(self, epoch=None):
        mae = self.mae.compute() * self.targets_std * 627.5
        rmse = self.rmse.compute() * self.targets_std * 627.5
        logging.info(f'{self.prefix} MAE: {mae}')
        logging.info(f'{self.prefix} RMSE: {rmse}')
        self.logger.log_metrics({f'{self.prefix} MAE': mae,
                                 f'{self.prefix} RMSE': rmse}, epoch)
        self.best_mae = min(self.best_mae, mae)
        self.best_rmse = min(self.best_rmse, rmse) 

    def on_fit_end(self):
        if self.best_mae != float('inf'):
            self.logger.log_metrics({f'{self.prefix} best MAE': self.best_mae})
        if self.best_rmse != float('inf'):
            self.logger.log_metrics({f'{self.prefix} best RMSE': self.best_rmse})


class TrIPLRSchedulerCallback(LRSchedulerCallback):
    def __init__(self, logger):
        super().__init__(logger)

    def get_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)