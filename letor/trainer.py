import sys
from numpy.random.mtrand import RandomState as RandomState
import torch
import torch.nn as nn
from torch.functional import Tensor
from experimaestro import Param, Config
from xpmir.learning.context import TrainerContext
from xpmir.letor.trainers.batchwise import (
    BatchwiseTrainer,
    BatchwiseLoss,
)
from xpmir.learning.context import Loss
from xpmir.rankers import LearnableScorer
from dataset.samplers import DistillationInBatchNegativesSampler
from dataset.records import ListwiseDistillationProductRecords

from xpmir.utils.utils import easylog

logger = easylog()

# --- Loss


class DistillationBatchwiseLoss(Config, nn.Module):
    """The abstract class for batchwise distillation loss"""

    weight: Param[float] = 1.0
    NAME = "?"

    def initialize(self, ranker: LearnableScorer):
        pass

    def process(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ):
        loss = self.compute(student_scores, teacher_scores, info)
        info.add_loss(Loss(f"batchwise-{self.NAME}", loss, self.weight))

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        """
        Compute the loss
        """
        raise NotImplementedError()


# KL distillation
class DistillationBatchwiseKLLoss(DistillationBatchwiseLoss):
    """
    Follow the code of the colbertv2 to do a KL distillation over
    a batch of 'negative' for each query
    """

    NAME = "Distil-Batch-KL"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ) -> torch.Tensor:
        log_teacher_scores = torch.nn.functional.log_softmax(teacher_scores, dim=-1)
        log_student_scores = torch.nn.functional.log_softmax(student_scores, dim=-1)
        return self.loss(log_student_scores, log_teacher_scores)


# The In Batch Negative Loss is defined in Xpmir

# --- Trainer
class DistillationInBatchNegativeTrainer(BatchwiseTrainer):

    sampler: Param[DistillationInBatchNegativesSampler]
    """A batch-wise sampler but contain"""

    lossfn: Param[BatchwiseLoss]
    """The In Batch Negative Loss"""

    lossfn_distillation: Param[DistillationBatchwiseLoss]
    """The distillation loss"""

    need_ibn: Param[bool] = True
    """Whether we apply ibn loss"""

    need_distil: Param[bool] = True
    """Whether we apply the distillation loss"""

    def __validate__(self) -> None:
        assert self.need_distil or self.need_ibn

    def initialize(self, random: RandomState, context: TrainerContext):
        super().initialize(random, context)
        self.lossfn.initialize(context)
        self.lossfn_distillation.initialize(self.ranker)

    def train_batch(self, batch: ListwiseDistillationProductRecords):
        # Get the next batch and compute the scores for each query/document
        # Get the training examples
        records = ListwiseDistillationProductRecords()
        records.add_topics(*batch.unique_topics)
        records.add_documents(*[d.document for d in batch.unique_documents])

        # Get the scores
        rel_scores = self.ranker(records, self.context)
        try:
            alpha = self.ranker.alpha
        except:
            alpha = 1

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # batch score shape [bs_q, bs_d] where bs_d = bs_q * nway and bs_q = bs
        batch_scores = rel_scores.reshape(
            len(batch.unique_queries), len(batch.unique_documents)
        )
        ibn_input = batch.ibn_relevance_view(batch_scores)  # shape [bs, (bs-1)*nway+1]
        ibn_target = batch.ibn_relevance().to(
            ibn_input.device
        )  # shape [bs, (bs-1)*nway+1]

        distil_input = batch.distillation_view(batch_scores)  # shape [bs, nway]
        distil_target = (
            torch.tensor([d.score for d in batch.unique_documents])
            .reshape_as(distil_input)
            .to(distil_input.device)
        )  # shape [bs, nway]

        if self.need_ibn:
            self.lossfn.process(ibn_input, ibn_target, self.context)
        if self.need_distil:
            self.lossfn_distillation.process(
                distil_input * alpha, distil_target, self.context
            )
