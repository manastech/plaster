"""
Jobs are:
    folders under ./jobs_folder
    Created by generators (pgen)
Runs are:
    Sub-folders of jobs
    Contain plaster_run.yaml files
    May contain plaster_output/ folders
"""

from munch import Munch
import time
import sys
import hashlib
import os
import uuid
from functools import cmp_to_key
from plumbum import cli, local, colors, FG
from plaster.tools.pipeline.pipeline import Pipeline
from plaster.tools.utils import utils
from plaster.tools.utils import tmp
from plaster.tools.schema import check
from plaster.tools.log.log import debug, important, colorful_exception, info
from plaster.tools.zap import zap
from plaster.run.call_bag import CallBag
from plaster.run.classify_rf.classify_rf_result import ClassifyRFResult
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.run.ims_import.ims_import_task import ImsImportTask
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sigproc_v1.sigproc_v1_result import SigprocV1Result
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result
from plaster.run.sigproc_v1.sigproc_v1_task import SigprocV1Task
from plaster.run.sigproc_v2.sigproc_v2_task import SigprocV2Task
from plaster.run.lnfit.lnfit_task import LNFitTask
from plaster.run.lnfit.lnfit_result import LNFitResult
from plaster.run.sim.sim_result import SimResult
from plaster.run.test_rf.test_rf_result import TestRFResult
from plaster.run.test_rf.test_rf_task import TestRFTask
from plaster.run.train_rf.train_rf_result import TrainRFResult
from plaster.run.train_rf.train_rf_task import TrainRFTask
from plaster.run.classify_rf.classify_rf_task import ClassifyRFTask
from plaster.run.prep.prep_task import PrepTask
from plaster.run.sim.sim_task import SimTask
from plaster.run.report.report_task import ReportTask
from plaster.run.survey_nn.survey_nn_task import SurveyNNTask
from plaster.run.survey_nn.survey_nn_result import SurveyNNResult
from plaster.run.test_nn.test_nn_task import TestNNTask
from plaster.run.test_nn.test_nn_result import TestNNResult


def find_run_folders(default=None, must_include_plaster_output=True):
    """
    This is run in the context of a notebook. If the calling notebook
    finds itself inside of a plaster_output folder then it knows to
    load the data folders from that location.

    If the plaster_output is not found, then it is running
    directly from the templates folder in projects/plaster/notebooks.
    In that case it expects to be given the default argument.
    """

    def sorted_subdirs_of(folder):
        return sorted(
            [
                sub_folder
                for sub_folder in folder.list()
                if (
                    sub_folder.is_dir()
                    and (
                        not must_include_plaster_output
                        or (sub_folder / "plaster_output").exists()
                    )
                )
            ]
        )

    if (local.cwd / "..").name == "plaster_output":
        # Search up to find a sweep_root if it exists
        search = local.cwd
        emergency_exit = 50
        while search != "/" and emergency_exit > 0:
            if (search / "sweep_root").exists() or (
                search / ".."
            ).name == "jobs_folder":
                return sorted_subdirs_of(search)
            search = search / ".."
            emergency_exit -= 1
        return [local.cwd / "../.."]
    else:
        assert default is not None
        return sorted_subdirs_of(
            local.path(local.env["ERISYON_ROOT"]) / "jobs_folder" / default
        )


def task_list_from_config(config):
    """
    Given a
    """
    task_map = dict(
        ims_import=ImsImportTask,
        sigproc_v1=SigprocV1Task,
        sigproc_v2=SigprocV2Task,
        lnfit=LNFitTask,
        prep=PrepTask,
        sim=SimTask,
        train_rf=TrainRFTask,
        test_rf=TestRFTask,
        classify_rf=ClassifyRFTask,
        survey_nn=SurveyNNTask,
        test_nn=TestNNTask,
        report=ReportTask,
    )

    tasks = []
    for task_name, task_info in config.items():
        if task_name.startswith("_"):
            continue
        elif task_name in task_map:
            task_klass = task_map[task_name]
        elif "task" in task_info:
            task_klass = task_map[task_info["task"]]
        else:
            raise ValueError(f"Unable to determine task type of {task_name}.")

        tasks += [(task_name, task_klass, task_info)]

    task_order = [
        ImsImportTask,
        SigprocV1Task,
        SigprocV2Task,
        LNFitTask,
        PrepTask,
        SimTask,
        SurveyNNTask,
        TrainRFTask,
        TestRFTask,
        ClassifyRFTask,
        TestNNTask,
        ReportTask,
    ]

    task_klass_to_order = {klass: i for i, klass in enumerate(task_order)}

    def task_order_compare(a, b):
        return (
            task_klass_to_order[a[1]] - task_klass_to_order[b[1]]
        )  # 1 to reference the klass (see tasks above)

    tasks = sorted(tasks, key=cmp_to_key(task_order_compare))

    return tasks


class RunPipeline(Pipeline):
    def _tmp_dir(self):
        """mock-point"""
        if local.path("./tmp_folder").exists():
            return local.path("./tmp_folder")
        else:
            return local.path(local.env["HOST_ERISYON_TMP"])

    def _translate_s3_references(self, task, skip_s3):
        """
        Any "inputs" block may have S3 references in which case
        plaster will do an s3 sync with that folder to a local cache
        and then substitute that local path so that Pipeline is always
        working with local files.
        """

        for input_name, src_path in dict(task.inputs or {}).items():
            if not input_name.startswith("_"):
                if src_path.startswith("s3:"):
                    if not skip_s3:
                        found_cache, dst_path = tmp.cache_path("plaster_s3", src_path)
                        if not found_cache:
                            important(f"Syncing from {src_path} to {dst_path}")
                            local["aws"]["s3", "sync", src_path, dst_path] & FG

                        # COPY the old src_path by prefixing with underscore
                        task.inputs["_" + input_name] = str(src_path)

                        # RESET the input to the new dst_path
                        task.inputs[input_name] = str(dst_path)

    def __init__(self, src_dir, dst_dir, config, **build_opts):
        src_dir = local.path(src_dir)
        dst_dir = local.path(dst_dir)

        skip_s3 = build_opts.pop("skip_s3", False)

        self._tasks = {}
        task_list = task_list_from_config(config)
        for task_name, task_klass, task_info in task_list:
            self._translate_s3_references(task_info, skip_s3)
            self._tasks[task_name] = (task_klass, task_info, {})

        n_fields_limit = build_opts.pop("n_fields_limit", None)
        if n_fields_limit is not None:
            important(f"Limiting to only {n_fields_limit} fields")

            # TASK: Convert these from named task to ANY task f that type
            if "ims_import" in self._tasks:
                self._tasks["ims_import"][1]["parameters"][
                    "n_fields_limit"
                ] = n_fields_limit

            if "sigproc_v1" in self._tasks:
                self._tasks["sigproc_v1"][1]["parameters"][
                    "n_fields_limit"
                ] = n_fields_limit

            if "sigproc_v2" in self._tasks:
                self._tasks["sigproc_v2"][1]["parameters"][
                    "n_fields_limit"
                ] = n_fields_limit

        super().__init__(src_dir, dst_dir, self._tasks, **build_opts)


class RunExecutor:
    def __init__(self, run_folder, tasks={}):
        self.run_folder = local.path(run_folder)
        self.yaml_path = self.run_folder / "plaster_run.yaml"
        self.dst_dir = self.run_folder / "plaster_output"
        self.tasks = tasks
        self.task_list = None

    def load(self):
        self.config = utils.yaml_load_munch(self.yaml_path)
        self.task_list = task_list_from_config(self.config)
        self.tasks = {
            task_name: task_info for task_name, task_klass, task_info in self.task_list
        }
        return self

    def save(self):
        self.run_folder.mkdir()
        utils.yaml_save(self.yaml_path, self.tasks)
        return self

    def execute(self, **kwargs):
        failure = None
        logs = []
        start_time = time.time()
        try:
            pipeline = RunPipeline(self.run_folder, self.dst_dir, self.tasks, **kwargs)
            logs = pipeline.logs()
        except Exception as e:
            failure = e

        if "_erisyon" in self.tasks:
            utils.yaml_save(str(self.run_folder / "_erisyon.yaml"), self.tasks._erisyon)

        # WRITE manifest
        utils.yaml_write(
            self.run_folder / "run_manifest.yaml",
            uuid=uuid.uuid4().hex,
            run_dir=str(self.run_folder),
            run_name=utils.block_search(self.config, "_erisyon.run_name", ""),
            elapsed_secs=time.time() - start_time,
            timestamp=time.time(),
            who=local.env.get("RUN_USER", "Unknown"),
            n_cpus=os.cpu_count(),
            tasks=self.tasks,
            kwargs=kwargs,
            cmdline_args=sys.argv,
            logs=logs,
            errors=dict(
                failure=f"{failure.__class__.__name__}: {failure}" if failure else None,
                pipeline_failures=[f"{i[1]}: {i[3]}" for i in logs if i[2] == "failed"],
            ),
        )

        if failure is not None:
            raise failure

        return pipeline.failed_count()


class RunResult:
    """
    Wraps results with handy task loading __getattr__ overload.
    Permits:

    run = RunResult(...)
    plaster.run.prep.*
    """

    default_result_klass_by_task_name = dict(
        classify_rf=ClassifyRFResult,
        ims_import=ImsImportResult,
        lnfit=LNFitResult,
        prep=PrepResult,
        sigproc_v1=SigprocV1Result,
        sigproc_v2=SigprocV2Result,
        sim=SimResult,
        test_nn=TestNNResult,
        test_rf=TestRFResult,
        train_rf=TrainRFResult,
        sim_nn=SimResult,  # Example of why it is needed to pull the klass from the run
        survey_nn=SurveyNNResult,
    )

    def __init__(self, run_folder, include_manifest=True):
        if run_folder.startswith("//"):
            run_folder = local.path(local.env["ERISYON_ROOT"]) / run_folder[2:]
        self.run_folder = local.path(run_folder)
        self.run_name = self.run_folder.name
        self.yaml_path = self.run_folder / "plaster_run.yaml"
        self.run_output_folder = self.run_folder / "plaster_output"
        self.inst_cache = {}
        self.store = utils.Store(str(self.run_folder / "_store.pkl"))

        self.config = utils.yaml_load_munch(self.yaml_path)

        self.result_klass_by_task_name = {
            task_name: RunResult.default_result_klass_by_task_name[
                task_config.get("task", task_name)
            ]
            for task_name, task_config in self.config.items()
            if not task_name.startswith("_")
        }

        if include_manifest:
            self.inst_cache["manifest"] = utils.yaml_load_munch(
                self.run_folder / "run_manifest.yaml"
            )
            if "_erisyon" in self.inst_cache["manifest"]["tasks"]:
                self.inst_cache["_erisyon"] = self.inst_cache["manifest"]["tasks"][
                    "_erisyon"
                ]
        else:
            _erisyon_cache = self.run_folder / "_erisyon.yaml"
            if _erisyon_cache.exists():
                self.inst_cache["_erisyon"] = utils.yaml_load_munch(_erisyon_cache)

    def set_cache(self, root_key, inst):
        self.inst_cache[root_key] = inst

    def has_result(self, task_name):
        """
        Handy helper to query for availability of result without raising
        """
        return task_name in self.result_klass_by_task_name.keys()

    def __getattr__(self, key):
        """
        Uses a cache and the dispatches to the klass to JIT load
        """
        inst = self.inst_cache.get(key)
        if inst is None:
            try:
                result_klass = self.result_klass_by_task_name[key]
            except KeyError:
                # There are attributes (eg: __getstate__) that are not
                # in the class AND are not result_klass_by_name so those
                # have to get converted into AttributeError exceptions
                raise AttributeError
            inst = result_klass(
                self.run_output_folder / key, is_loaded_result=True, run=self
            )
            self.inst_cache[key] = inst
        return inst

    def __getitem__(self, item):
        return getattr(self, item)

    # Note, the following helpers are here because they involve
    # more than one sub-task. For example, test_rf_call_bag needs
    # the prep_result.
    #
    # Our pattern is to keep lower-level objects (in this case test_rf)
    # from having to know things about the other modules (prep_result in this case).

    def get_available_classifiers(self):
        # Add to these if there are others available, or change the order to
        # determine which is used by default if more than one is available in
        # a RunResult.
        supported_classifiers_by_preference = ["rf", "nn"]
        available_classifiers_by_preference = []
        for c in supported_classifiers_by_preference:
            if self.has_result(f"test_{c}"):
                available_classifiers_by_preference += [c]
        return available_classifiers_by_preference

    def test_call_bag(self, classifier=None, use_train_data=False):
        """
        Get a CallBag for a specific classifier, or a preferred available
        classifier if classifier is None.

        classifier: None, or some name from supported_classifiers_by_preference
        """
        if classifier is None:
            classifier = self.get_available_classifiers()[0]
        return self[f"test_{classifier.lower()}_call_bag"](
            use_train_data=use_train_data
        )

    def test_rf_call_bag(self, use_train_data=False):
        """
        Get a CallBag for the RF classifier on this plaster.run.
        use_train_data=True when you want to look at over-fitting.
        """
        if use_train_data:
            true_pep_iz = self.test_rf.train_true_pep_iz
            pred_pep_iz = self.test_rf.train_pred_pep_iz
            scores = self.test_rf.train_scores
            all_class_scores = self.test_rf.train_all_class_scores
            cached_pr = self.test_rf.train_peps_pr
            cached_pr_abund = self.test_rf.train_peps_pr_abund
            check.affirm(
                true_pep_iz is not None and pred_pep_iz is not None,
                "The test_rf task was not run with the training_set",
            )
        else:
            true_pep_iz = self.test_rf.test_true_pep_iz
            pred_pep_iz = self.test_rf.test_pred_pep_iz
            scores = self.test_rf.test_scores
            all_class_scores = self.test_rf.test_all_class_scores
            cached_pr = self.test_rf.test_peps_pr
            cached_pr_abund = self.test_rf.test_peps_pr_abund

        return CallBag(
            true_pep_iz=true_pep_iz,
            pred_pep_iz=pred_pep_iz,
            scores=scores,
            all_class_scores=all_class_scores,
            prep_result=self.prep,
            sim_result=self.sim,
            cached_pr=cached_pr,
            cached_pr_abund=cached_pr_abund,
            classifier_name="rf",
        )

    def classify_rf_call_bag(self):
        """
        Get a call bag for classification of real sigprocv2 data.
        """
        return CallBag(
            pred_pep_iz=self.classify_rf.pred_pep_iz,
            scores=self.classify_rf.scores,
            prep_result=self.prep,
            classifier_name="classify_rf",
        )

    def test_nn_call_bag(self, use_train_data=False):
        """
        Get a CallBag for the NN classifier on this plaster.run.
        use_train_data=True when you want to look at over-fitting.
        """
        if use_train_data:
            true_pep_iz = self.test_nn.train_true_pep_iz
            pred_pep_iz = self.test_nn.train_pred_pep_iz
            check.affirm(
                true_pep_iz is not None and pred_pep_iz is not None,
                "The test_nn task was not run with the training_set",
            )
            cached_pr = self.test_nn.train_peps_pr
        else:
            true_pep_iz = self.test_nn.test_true_pep_iz
            pred_pep_iz = self.test_nn.test_pred_pep_iz
            cached_pr = self.test_nn.test_peps_pr

        return CallBag(
            true_pep_iz=true_pep_iz,
            pred_pep_iz=pred_pep_iz,
            scores=self.test_nn.test_scores,
            prep_result=self.prep,
            sim_result=self.sim,
            cached_pr=cached_pr,
            classifier_name="nn",
        )

    def peps_prs_report_df(
        self,
        include_decoys=False,
        in_report_only=False,
        ptm_peps_only=False,
        force_compute_prs=False,
        pr_with_abundance=False,
        classifier=None,
    ):
        """
        Get a df that contains a variety of attributes for peptides used in reporting.

        This is a helper in run because it requires information from prep, sim, and classify.

        in_report: only consider peptides that are part of "proteins of interest"
        ptms_only: only include peptides which contain user-specified ptm locations.
        """
        if pr_with_abundance and self.prep.pros_abundance() is None:
            return None

        cb = self.test_call_bag(classifier=classifier)
        peps_ptms = cb.peps__pepstrs__flustrs__p2(
            include_decoys=include_decoys,
            in_report_only=in_report_only,
            ptm_peps_only=ptm_peps_only,
        )
        pep_iz = peps_ptms.pep_i.unique()
        if pr_with_abundance:
            prs = cb.pr_curve_by_pep_with_abundance(
                pep_iz=pep_iz, force_compute=force_compute_prs
            )
        else:
            prs = cb.pr_curve_by_pep(pep_iz=pep_iz, force_compute=force_compute_prs)
        return peps_ptms.set_index("pep_i").join(prs.set_index("pep_i")).reset_index()


def _do_store_get_cache_or_execute(
    run=None, key=None, inner_fn=None, _args=None, _clear_cache=False
):
    """
    If the key is in the run's store, return that. Otherwise execute fn and store.
    """
    check.t(run, RunResult)
    check.t(key, str)
    if _clear_cache:
        plaster.run.store.rm(key)
    if key not in plaster.run.store:
        return False, inner_fn(*_args)
    else:
        return True, plaster.run.store[key]


def pmap_runstore(fn, work_orders, _clear_cache=False, **kws):
    """
    Parallel run fn over the work_orders.
    Arguments:
        work_orders: a list of dicts.
            Each work_order dict MUST contain a 'run', 'key', and 'args' parameters that are used
            to update the appropriate run's store with that key.
    """
    work_orders = [
        dict(
            **wo,
            fn=_do_store_get_cache_or_execute,
            inner_fn=fn,
            _clear_cache=_clear_cache,
        )
        for wo in work_orders
    ]

    p = zap.work_orders(work_orders, **kws)
    # UPDATE stores. This is done in the master process to avoid sync issues
    for wo, result in zip(work_orders, p.results):
        from_cache, result = result
        if not from_cache:
            wo["run"].store[wo["key"]] = result
