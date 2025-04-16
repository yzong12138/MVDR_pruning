from typing import List
from functools import cached_property
from xpmir.papers import configuration
from xpmir.learning.optim import (
    get_optimizers,
    ParameterOptimizer,
    RegexParameterFilter,
    AdamW,
    Adam,
    Adafactor,
    SGD,
)
from xpmir.learning.schedulers import LinearWithWarmup
from xpmir.papers.helpers.optim import TransformerOptimization


@configuration()
class ColBERTOptimization(TransformerOptimization):
    """This optimization propose a optimizer with a different lr for the
    alpha"""

    alpha_lr: float = 5.0e-3
    """The learning rate for the alpha"""

    alpha_param_name: List[str] = ["alpha"]

    def get_optimizer(self, regularization, lr):
        # Set weight decay to 0 if no regularization
        weight_decay = self.weight_decay if regularization else 0

        if self.optimizer_name == "adam-w":
            return AdamW(
                lr=lr,
                weight_decay=weight_decay,
                eps=self.eps,
            )
        elif self.optimizer_name == "adam":
            return Adam(lr, weight_decay=weight_decay, eps=self.eps)
        elif self.optimizer_name == "sgd":
            return SGD(lr=lr, weight_decay=weight_decay)
        elif self.optimizer_name == "adafactor":
            return Adafactor(lr=lr, weight_decay=weight_decay, relative_step=lr is None)
        else:
            raise ValueError(f"Cannot handle optimizer named {self.optimizer_name}")

    @cached_property
    def scheduler_instance(self):
        scheduler = (
            LinearWithWarmup(
                num_warmup_steps=self.num_warmup_steps,
                min_factor=self.warmup_min_factor,
            )
            if self.scheduler
            else None
        )
        return scheduler

    @cached_property
    def optimizer(self):
        if not self.re_no_l2_regularization:
            return get_optimizers(
                [
                    ParameterOptimizer(
                        scheduler=self.scheduler_instance,
                        optimizer=self.get_optimizer(False, self.alpha_lr),
                        filter=RegexParameterFilter(includes=self.alpha_param_name),
                    ),
                    ParameterOptimizer(
                        scheduler=self.scheduler_instance,
                        optimizer=self.get_optimizer(True, self.lr),
                    ),
                ]
            )

        return get_optimizers(
            [
                ParameterOptimizer(
                    scheduler=self.scheduler_instance,
                    optimizer=self.get_optimizer(False, self.alpha_lr),
                    filter=RegexParameterFilter(includes=self.alpha_param_name),
                ),
                ParameterOptimizer(
                    scheduler=self.scheduler_instance,
                    optimizer=self.get_optimizer(False, self.lr),
                    filter=RegexParameterFilter(includes=self.re_no_l2_regularization),
                ),
                ParameterOptimizer(
                    scheduler=self.scheduler_instance,
                    optimizer=self.get_optimizer(True, self.lr),
                ),
            ]
        )
