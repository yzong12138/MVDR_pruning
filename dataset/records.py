import torch
from xpmir.letor.records import ProductRecords


class ListwiseDistillationProductRecords(ProductRecords):
    def ibn_relevance(self):
        """Return the target relevance for the ibn scenario
        shape [bs, (bs-1)*nway+1]"""
        bs = len(self._topics)
        nway = int(len(self._documents) / bs)

        # return shape [bs, (bs-1)*nway+1]
        return torch.block_diag(
            *torch.cat((torch.ones(1), torch.zeros(nway - 1)))
            .unsqueeze(0)
            .repeat(bs, 1)
        )[:, : -(nway - 1)]

    def ibn_relevance_view(self, input_scores):
        """
        input the scores of shape [bs, bs*nway]
        return the scores of shape [bs, (bs-1)*nway+1]
        """
        bs = len(self._topics)
        nway = int(len(self._documents) / bs)

        keep = torch.cat(
            [torch.arange(1)]
            + [
                torch.arange(
                    nway + nway * (bs + 1) * qidx, nway * (bs + 1) * (qidx + 1) + 1
                )
                for qidx in range(bs - 1)
            ]
        )
        return input_scores.reshape(-1)[keep].reshape(bs, -1)

    def distillation_view(self, input_scores):
        """return the distillation scores to be considered
        shape [bs, nway], the first column correspond to positive"""
        bs = len(self._topics)
        nway = int(len(self._documents) / bs)
        keep = torch.cat(
            [
                torch.arange(qidx * nway * (bs + 1), qidx * nway * (bs + 1) + nway)
                for qidx in range(bs)
            ]
        )
        return input_scores.reshape(-1)[keep].reshape(bs, -1)

    def __getitem__(self, ix: slice | int):
        nway = int(len(self._documents) / len(self._topics))
        if isinstance(ix, slice):
            records = ListwiseDistillationProductRecords()
            for i in range(ix.start, min(ix.stop, len(self._topics)), ix.step or 1):
                records.add_topics(self._topics[i])
                records.add_documents(*self._documents[i * nway : (i + 1) * nway])
            return records

        records = ListwiseDistillationProductRecords()
        records.add_topics(self._topics[ix])
        records.add_documents(*self._documents[ix * nway : (ix + 1) * nway])
