from zest import zest
import numpy as np
import pandas as pd
from plaster.run.prep.prep_result import PrepResult
from plaster.run.prep.prep_params import PrepParams

# Hint: use stub_prep_result

# @zest.skip("m", "Manas")
def zest_PrepResult():

    def pros():
        result = None
        default_params = None
        params_with_abundance = None

        def _before():
            nonlocal result, default_params, params_with_abundance

            default_params = PrepParams(proteins=[
                dict(name="id_0", sequence="ABC"),
                dict(name="id_1", sequence="CCE"),
                dict(name="id_2", sequence="AAB"),
            ])

            params_with_abundance = PrepParams(proteins=[
                dict(name="id_0", sequence="ABC", abundance=2),
                dict(name="id_1", sequence="CCE"),
                dict(name="id_2", sequence="AAB", abundance=1),
            ])

            result = PrepResult.stub_prep_result(
                pros=[".", "ABCDEFGHI", "DDD"],
                pro_is_decoys=[False, False, True],
                peps=[".", "AAA", "DDD"],
                pep_pro_iz=[0, 1, 2],
            )

            result.params = default_params

        def it_gets_a_dataframe_with_pro_id_index():
            pros = result.pros()
            assert isinstance(pros, pd.DataFrame)
            assert "pro_id" in pros.columns

        def it_includes_abundances_if_in_params_proteins():
            assert "abundance" not in result.pros().columns
            result.params = params_with_abundance
            assert "abundance" in result.pros().columns

        def it_gets_n_pros():
            assert result.n_pros == 3

        def it_gets_pros_abundance():
            result.params = params_with_abundance
            pros_abundance = result.pros_abundance()
            assert len(pros_abundance) == result.n_pros

            def it_converts_nans_to_zero():
                nonlocal pros_abundance
                assert np.any([np.isnan(x) for x in result.pros().abundance])
                assert np.all([not np.isnan(x) for x in pros_abundance])

            zest()

        def it_gets_pros__ptm_locs():
            raise NotImplementedError

        def it_gets_pros__from_decoys():
            raise NotImplementedError

        def it_gets_pros__no_decoys():
            raise NotImplementedError

        def it_gets_proseqs():
            raise NotImplementedError

        def it_gets_prostrs():
            raise NotImplementedError

        zest()

    def pros_of_interest():
        def it_sets_pros_of_interest():
            raise NotImplementedError

        def it_gets_n_pros_of_interest():
            raise NotImplementedError

        def it_asserts_protein_ids():
            raise NotImplementedError

        def it_sets_pro_ptm_locs():
            raise NotImplementedError

        def it_gets_pro_ptm_locs():
            raise NotImplementedError

        zest()

    def peps():
        def it_gets_peps():
            raise NotImplementedError

        def it_gets_n_peps():
            raise NotImplementedError

        def it_gets_pepstrs():
            raise NotImplementedError

        def it_gets_pepseqs():
            raise NotImplementedError

        def it_gets_peps_abundance():
            raise NotImplementedError

            def it_fills_zero_for_missing_abundance():
                raise NotImplementedError

            zest()

        def it_gets_peps__no_decoys():
            raise NotImplementedError

            def it_handles_empty_return():
                raise NotImplementedError

            zest()

        def it_gets_peps__from_decoys():
            raise NotImplementedError

            def it_handles_empty_return():
                raise NotImplementedError

            zest()

        def peps__in_report():
            raise NotImplementedError

            def it_handles_empty_return():
                raise NotImplementedError

            zest()

        def it_gets_peps__pepseqs():
            raise NotImplementedError

        def it_gets_pepseqs__no_decoys():
            raise NotImplementedError

        def peps__ptms():
            def it_filters_decoys():
                raise NotImplementedError

            def it_filters_in_report_only():
                raise NotImplementedError

            def it_filters_ptm_peps_only():
                raise NotImplementedError

            def it_ptms_to_rows():
                raise NotImplementedError

            zest()

        zest()

    def pros_and_peps():
        def it_gets_pros__peps():
            raise NotImplementedError

        def it_gets_pros__peps__pepstrs():
            raise NotImplementedError

        zest()

    zest()
