from typing import List, Iterator, NamedTuple, Iterable
import json
import torch
from experimaestro import Config, Param
from datamaestro.data import File
from datamaestro_text.data.ir.base import (
    TopicRecord,
    DocumentRecord,
    ScoredItem,
    IDItem,
    create_record,
)

from xpmir.letor.samplers.hydrators import SampleHydrator
from xpmir.rankers import ScoredDocument
from xpmir.utils.iter import (
    SkippingIterator,
    SerializableIteratorTransform,
)


class ListwiseDistillationSample(NamedTuple):
    query: TopicRecord
    """The query"""

    documents: List[DocumentRecord]
    """List of documents with teacher scores"""


class ListwiseDistillationSamples(Config, Iterable[ListwiseDistillationSample]):
    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        raise NotImplementedError()

    @property
    def nway(self):
        raise NotImplementedError()


class ListwiseHydrator(ListwiseDistillationSamples, SampleHydrator):
    """Hydrate ID-based samples with document and/or query content"""

    samples: Param[ListwiseDistillationSamples]
    """The distillation samples without texts for query and documents"""

    def transform(self, sample: ListwiseDistillationSample):
        topic, documents = sample.query, sample.documents

        if transformed := self.transform_topics([topic]):
            topic = transformed[0]

        if transformed := self.transform_documents(documents):
            documents = tuple(
                ScoredDocument(d, sd[ScoredItem].score)
                for d, sd in zip(transformed, documents)
            )

        return ListwiseDistillationSample(topic, documents)

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        iterator = iter(self.samples)
        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform
        )

    @property
    def nway(self):
        return self.samples.nway


class JSONBasedBatchDistillationSamples(ListwiseDistillationSamples, File):
    """A JSON based batchwise distillation samples dataset
    current moment, it contains the ids. The ids are of type int, but they are
    external ids when transform to string
    """

    nway_num: Param[int] = 64
    """the number of distillation samples to use for each query"""

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        return self.iter()

    def iter(self) -> Iterator[ListwiseDistillationSample]:
        def iterate():
            with self.path.open("rt") as fp:
                for line in fp:
                    sample = json.loads(line)
                    query = create_record(id=str(sample[0]))

                    documents = []
                    documents.append(  # append the positive
                        DocumentRecord(
                            IDItem(str(sample[1][0])), ScoredItem(float(sample[1][1]))
                        )
                    )
                    num_neg_samples = min(len(sample) - 1, self.nway) - 1
                    indices = torch.randperm(len(sample) - 2) + 2
                    for i, _ in zip(indices, range(num_neg_samples)):
                        documents.append(
                            DocumentRecord(
                                IDItem(str(sample[i][0])),
                                ScoredItem(float(sample[i][1])),
                            )
                        )
                    yield ListwiseDistillationSample(query, documents)

        return SkippingIterator(iterate())

    @property
    def nway(self):
        return self.nway_num
