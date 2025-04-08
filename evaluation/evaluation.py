import sys
from itertools import chain
import pandas as pd
from experimaestro import tags
from datamaestro import prepare_dataset
from xpmir.utils.functools import cache
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.papers.helpers.samplers import prepare_collection, MEASURES
from xpmir.rankers import Retriever
from evaluation.cache import DataCachedEvaluate
from xpmir.utils.logging import easylog

logger = easylog()


class TwoStageEvaluationsCollection(EvaluationsCollection):
    """A dataset which need to be evaluated in two stage"""

    def evaluate_first_stage_retriever(
        self, retriever, launcher=None, model_id=None, overwrite=False, init_tasks=[]
    ):
        """Evaluate a retriever for all the evaluations in this collection (the
        tasks are submitted to the experimaestro scheduler)
        It will cache the retrieved results to accelerate the whole pipeline
        """
        if model_id is not None and not overwrite:
            assert (
                model_id not in self.per_model
            ), f"Model with ID `{model_id}` was already evaluated"

        results = []
        cached_retrievers = []
        for key, evaluations in self.collection.items():
            cached_retriever, result = evaluations.evaluate_retriever(
                retriever, launcher, init_tasks=init_tasks, first_stage=True
            )
            results.append((key, result))
            cached_retrievers.append(cached_retriever)

        # Adds to per model results
        if model_id is not None:
            self.per_model[model_id] = results

        return results, cached_retrievers

    def evaluate_second_stage_retriever(
        self,
        retrievers,
        launcher=None,
        model_id=None,
        overwrite=False,
        init_tasks=[],
    ):
        """Evaluate a retriever for all the evaluations in this collection (the
        tasks are submitted to the experimaestro scheduler)
        """
        if model_id is not None and not overwrite:
            assert (
                model_id not in self.per_model
            ), f"Model with ID `{model_id}` was already evaluated"

        results = []

        for (key, evaluations), retriever in zip(self.collection.items(), retrievers):
            _retriever, result = evaluations.evaluate_retriever(
                retriever, launcher, init_tasks=init_tasks, first_stage=False
            )
            results.append((key, result))

        # Adds to per model results
        if model_id is not None:
            self.per_model[model_id] = results

        return results

    def output_results_per_tag(self, file=sys.stdout, printed_metric=None):
        """Outputs the results for each collection, based on the retriever tags
        to build the table
        """
        # Loop over all collections
        for key, evaluations in self.collection.items():
            print(f"## Dataset {key}\n", file=file)  # noqa: T201
            evaluations.output_results_per_tag(file, printed_metric)
            print("\n", file=file)  # noqa: T201

    def output_detailed_path(self):
        # only output the detailed path of the model
        detailed_paths = {}
        for key, evaluations in self.collection.items():
            detailed_paths.update(evaluations.print_detailed_path_with_cp_tags())
        return detailed_paths


class TwoStageEvaluations(Evaluations):
    def print_detailed_path_with_cp_tags(self):
        """Only print the information with certain tags"""
        return {
            tags: adhoc_res.detailed
            for tags, adhoc_res in self.per_tags.items()
            if "cp" in tags.keys()  # avoid the evaluation of the first stage
        }

    def output_results_per_tag(self, file=sys.stdout, printed_metric=None):
        return self.to_dataframe(printed_metric).to_markdown(file)

    def evaluate_retriever(
        self, retriever, launcher=None, init_tasks=[], first_stage=False
    ):
        """Evaluate the retriver"""
        if not first_stage:
            return super().evaluate_retriever(retriever, launcher, init_tasks)
        else:
            if not isinstance(retriever, Retriever):
                retriever = retriever(self.dataset.documents)

            data_cached_evalutation = DataCachedEvaluate(
                retriever=retriever,
                measures=self.measures,
                dataset=self.dataset,
                topic_wrapper=self.topic_wrapper,
            ).submit(launcher=launcher, init_tasks=init_tasks)
            results = data_cached_evalutation.results
            cached_retriever = data_cached_evalutation.cached_retriever

            self.add(results)
            # Use retriever tags
            retriever_tags = tags(results)
            if retriever_tags:
                self.per_tags[retriever_tags] = results

            return cached_retriever, results

    def to_dataframe(self, printed_metric=None) -> pd.DataFrame:
        # Get all the tags
        tags = list(
            set(chain(*[tags_dict.keys() for tags_dict in self.per_tags.keys()]))
        )
        tags.sort()

        # Get all the results and metrics
        to_process = []
        metrics = set()
        for tags_dict, evaluate in self.per_tags.items():
            try:
                results = evaluate.get_results()
                if printed_metric:
                    metrics.update(printed_metric)
                else:
                    metrics.update(results.keys())
                to_process.append((tags_dict, results))
            except FileNotFoundError:
                logger.error("Cannot retrieve evaluation results for %s", tags_dict)

        # Sort metrics
        metrics = list(metrics)
        metrics.sort()

        # Table header
        columns = []
        for tag in tags:
            columns.append(["tag", tag])
        for metric in metrics:
            columns.append(["m", metric])

        # Output the results
        rows = []
        for tags_dict, results in to_process:
            row = []
            # tag values
            for k in tags:
                row.append(str(tags_dict.get(k, "")))

            # metric values
            for metric in metrics:
                row.append(results.get(metric, ""))
            rows.append(row)

        index = pd.MultiIndex.from_tuples(columns)
        return pd.DataFrame(rows, columns=index)


@cache
def msmarco_v1_tests():
    return TwoStageEvaluationsCollection(
        msmarco_dev=TwoStageEvaluations(
            prepare_collection("irds.msmarco-passage.dev.small").tag("ds", "dev_small"),
            MEASURES,
        ),
        trec2019=TwoStageEvaluations(
            prepare_dataset("irds.msmarco-passage.trec-dl-2019.judged").tag(
                "ds", "trec2019"
            ),
            MEASURES,
        ),
        trec2020=TwoStageEvaluations(
            prepare_dataset("irds.msmarco-passage.trec-dl-2020.judged").tag(
                "ds", "trec2020"
            ),
            MEASURES,
        ),
    )


@cache
def OoD_tests(beir_lotte_set):
    """The beir test_set"""
    return TwoStageEvaluationsCollection(
        OoD_subset=TwoStageEvaluations(prepare_dataset(beir_lotte_set), MEASURES)
    )
