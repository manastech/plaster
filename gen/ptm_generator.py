import re
from munch import Munch
from plaster.gen.base_generator import BaseGenerator
from plaster.gen import task_templates
from plaster.tools.schema.schema import Schema as s
from plaster.tools.utils import utils
from plaster.tools.log.log import debug


class PTMGenerator(BaseGenerator):
    """
    Use one set of labels to identify peptides and another label
    to measure quantities of PTM forms.

    Assumptions:
    * Only one label_set channel has PTMs in it.

    Generator-specific arguments:
    @--ptm-peptide="P10000"                       # Peptide to examine; Required and Repeatable

    """

    schema = s(
        s.is_kws_r(
            **BaseGenerator.job_setup_schema.schema(),
            **BaseGenerator.protein_schema.schema(),
            **BaseGenerator.label_set_schema.schema(),
            **BaseGenerator.scope_run_schema.schema(),
            **BaseGenerator.peptide_setup_schema.schema(),
            **BaseGenerator.error_model_schema.schema(),
            ptm_protein_of_interest=s.is_list(
                s.is_str(allow_empty_string=False),
                help="The name of the protein to look for PTMs",
            ),
            ptm_label=s.is_str(allow_empty_string=False, help="The PTM label"),
            n_peptides_limit=s.is_int(
                noneable=True, help="Useful for debugging to limit peptide counts"
            ),
        )
    )

    defaults = Munch(
        n_edmans=10,
        n_pres=1,
        n_mocks=0,
        decoys="none",
        random_seed=None,
        ptm_label="S[p]T[p]",
        dye_beta=[7500.0],
        dye_sigma=[0.16],
    )

    def apply_defaults(self):
        super().apply_defaults()

        # Plumbum creates empty lists on list switches. This means
        # that the apply defaults doesn't quite work right.
        # TASK: Find a cleaner solution. For now hard-code
        if len(self.dye_beta) == 0:
            self.dye_beta = self.defaults.dye_beta
        if len(self.dye_sigma) == 0:
            self.dye_sigma = self.defaults.dye_sigma

    def generate(self):
        runs = []
        for protease, aa_list, err_set in self.run_parameter_permutator():

            # GENERATE e-block
            e_block = self.erisyon_block(aa_list, protease, err_set)

            ptm_labels = re.compile(r"[A-Z]\[.\]", re.IGNORECASE).findall(
                self.ptm_label
            )

            # This feels a likely hacky
            ptm_aas = "".join([i[0] for i in ptm_labels])
            if ptm_aas not in aa_list:
                aa_list = tuple(list(aa_list) + [ptm_aas])

            # GENERATE the usual non-ptm prep, sim, train
            prep_task = task_templates.prep(
                self.protein,
                protease,
                self.decoys,
                n_peptides_limit=self.n_peptides_limit,
                proteins_of_interest=self.proteins_of_interest,
            )

            sim_task = task_templates.sim(
                list(aa_list),
                n_pres=self.n_pres,
                n_mocks=self.n_mocks,
                n_edmans=self.n_edmans,
                dye_beta=self.dye_beta,
                dye_sigma=self.dye_sigma,
                ptm_labels=ptm_labels,
            )

            train_task = task_templates.train_rf()

            # GENERATE the ptm tasks
            ptm_train_rf_task = task_templates.ptm_train_rf(
                ptm_labels, self.ptm_protein_of_interest
            )

            ptm_classify_test_rf_task = task_templates.ptm_classify_test_rf()

            # CREATE the run
            run = Munch(
                run_name=self.run_name(aa_list, protease, err_set),
                **e_block,
                **prep_task,
                **sim_task,
                **train_task,
                **ptm_train_rf_task,
                **ptm_classify_test_rf_task,
            )
            runs += [run]

        self.report_section_run_array(runs, to_load=["plaster", "sim", "prep", "ptm"])
        self.report_section_from_template("ptm_template.ipynb")

        n_runs = len(runs)
        self.report_preamble(
            utils.smart_wrap(
                f"""
                # PTM Report
                ## {n_runs} run(s) processed.
                """
            )
        )

        return runs
