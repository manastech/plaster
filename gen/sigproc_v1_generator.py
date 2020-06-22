from munch import Munch
from plaster.gen.base_generator import BaseGenerator
from plaster.tools.schema.schema import Schema as s
from plaster.tools.utils import utils
from plaster.tools.calibration.calibration import Calibration
from plumbum import local


class SigprocV1Generator(BaseGenerator):
    """
    Examine sigprocv2(s) and study their results
    """

    schema = s(
        s.is_kws_r(
            **BaseGenerator.job_setup_schema.schema(),
            **BaseGenerator.lnfit_schema.schema(),
            **BaseGenerator.sigproc_source_schema.schema(),
            **BaseGenerator.sigproc_v1_schema.schema(),
        )
    )

    defaults = Munch(
        lnfit_name=None,
        lnfit_params=None,
        lnfit_dye_on_threshold=None,
        lnfit_photometry_only=None,
        movie=False,
        n_frames_limit=None,
        radial_filter=None,
        peak_find_n_cycles=4,
        peak_find_start=0,
        anomaly_iqr_cutoff=95,
    )

    def generate(self):
        run_descs = []
        sigproc_tasks = self.sigprocs_v1()
        if len(sigproc_tasks) == 0:
            raise ValueError(
                "No sigprocv2 tasks were found. This might be due to an empty block of another switch."
            )

        for sigproc_i, sigproc_task in enumerate(sigproc_tasks):
            lnfit_tasks = self.lnfits()

            sigproc_source = ""
            for k, v in sigproc_task.items():
                if "ims_import" in k:
                    sigproc_source = local.path(v.inputs.src_dir).name
                    break

            run_name = f"sigproc_v1_{sigproc_i}_{sigproc_source}"
            if self.force_run_name is not None:
                run_name = self.force_run_name

            run_desc = Munch(run_name=run_name, **sigproc_task, **lnfit_tasks,)

            sigproc_template = "sigproc_v1_template.ipynb"
            if self.movie:
                sigproc_template = "sigproc_v1_movie_template.ipynb"

            self.report_section_markdown(f"# RUN {run_desc.run_name}\n")
            self.report_section_run_object(run_desc)

            self.report_section_from_template(sigproc_template)
            if lnfit_tasks:
                self.report_section_from_template("lnfit_template.ipynb")

            run_descs += [run_desc]

        n_run_descs = len(run_descs)
        self.report_preamble(
            utils.smart_wrap(
                f"""
                # Signal Processing Overview
                ## {n_run_descs} run(s) processed.
            """
            )
        )

        return run_descs
