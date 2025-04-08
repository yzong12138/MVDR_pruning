from typing import Dict, NamedTuple
import json
from experimaestro import Param, Annotated, pathgenerator
from pathlib import Path
from xpmir.evaluation import Evaluate, get_run
from xpmir.rankers import Retriever, ScoredDocument
from datamaestro_text.data.ir import IDItem
from datamaestro_text.data.ir.trec import TrecAdhocResults


class DataCachedRetriever(Retriever):
    """
    A retriever which read the results from a given path instead of
    re-run the retriever using the model
    """

    path: Param[Path]
    """The path of the datas"""

    retrieved: Dict
    """The dictionary of the retrieved documents"""

    def initialize(self):
        super().initialize()
        with open(self.path, "rt") as fp:
            line = next(fp)
            self.retrieved = json.loads(line)

    def retrieve(self, record):
        documents_cached = self.retrieved.get(record[IDItem].id, None)
        if documents_cached is None:
            raise RuntimeError(f"q_id not found: {record[IDItem].id}")
        documents = self.store.documents_ext(list(documents_cached.keys()))
        scores = list(documents_cached.values())
        return [ScoredDocument(d, s) for d, s in zip(documents, scores)]


class TwoStageEvaluateOutput(NamedTuple):
    """The data structure for the output of a learner. It contains a dictionary
    where the key is the name of the listener and the value is the output of
    that listener"""

    results: TrecAdhocResults

    cached_retriever: DataCachedRetriever


class DataCachedEvaluate(Evaluate):
    """This retriever takes a dataset and and a retriever and do the evalution
    It cached the retrieved first k documents' ids based on the query ids.
    It can be used as a first-stage reranker to save the time for the second stage
    retrieving
    """

    cached: Annotated[Path, pathgenerator("cached.json")]
    """Path for detailed results"""

    def task_outputs(self, dep):
        return TwoStageEvaluateOutput(
            results=dep(
                TrecAdhocResults(
                    id="",
                    results=self.aggregated,
                    detailed=self.detailed,
                    metrics=self.measures,
                )
            ),
            cached_retriever=dep(
                DataCachedRetriever(
                    path=self.cached,
                    store=self.dataset.documents,
                )
            ),
        )

    def execute(self):
        self.retriever.initialize()
        run = get_run(self.retriever, self.dataset)
        # write down the runs in the document
        self.caching(run)
        self._execute(run, self.dataset.assessments)

    def caching(self, run):
        with open(self.cached, "w") as outfile:
            json.dump(run, outfile)
