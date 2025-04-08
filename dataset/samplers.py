from typing import Any
from experimaestro import Param

from xpmir.learning.base import Sampler
from xpmir.letor.samplers import BatchwiseSampler
from dataset.records import ListwiseDistillationProductRecords
from dataset.distillation import ListwiseDistillationSamples, ListwiseDistillationSample

from xpmir.utils.iter import (
    SerializableIterator,
    SkippingIterator,
    SerializableIteratorAdapter,
)

from xpmir.letor.records import BatchwiseRecords


class DistillationListwiseSampler(Sampler):
    """Abstract class for pairwise samplers which output a set of (query,
    positive, and another list of document)"""

    samples: Param[ListwiseDistillationSamples]
    """the distillation samples"""

    def listwise_sampler(self) -> SerializableIterator[ListwiseDistillationSample, Any]:
        return SkippingIterator.make_serializable(iter(self.samples))


class DistillationInBatchNegativesSampler(BatchwiseSampler):
    """An in-batch negative sampler constructured from a listwise one,
    we consider only the negatives from the other queries"""

    sampler: Param[DistillationListwiseSampler]
    """The base listwise sampler"""

    def initialize(self, random):
        super().initialize(random)
        self.sampler.initialize(random)

    def batchwise_iter(
        self, batch_size: int
    ) -> SerializableIterator[BatchwiseRecords, Any]:
        def iter(list_iter):
            while True:
                batch = ListwiseDistillationProductRecords()
                for _, record in zip(range(batch_size), list_iter):
                    batch.add_topics(record.query)
                    batch.add_documents(*record.documents)
                yield batch

        return SerializableIteratorAdapter(self.sampler.listwise_sampler(), iter)
