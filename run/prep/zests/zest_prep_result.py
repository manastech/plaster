from zest import zest
import itertools
import numpy as np
import pandas as pd
from plaster.run.prep.prep_result import PrepResult
from plaster.run.prep.prep_params import PrepParams


def stub_prep_params(pros, pro_abundances=[]):
    def stub_protein(i, seq, abundance):
        pro = dict(name=f"id_{i}", sequence=seq)
        if abundance is not None:
            pro["abundance"] = abundance
        return pro

    proteins = [stub_protein(i, seq, abundance)
                for i, (seq, abundance) in enumerate(itertools.zip_longest(pros, pro_abundances))]

    return PrepParams(proteins=proteins)


def zest_PrepResult():

    def pros():
        result = None
        default_params = None
        params_with_abundance = None

        def _before():
            nonlocal result, default_params, params_with_abundance

            default_params = stub_prep_params(pros=["ABC", "CCE", "AAB"])
            params_with_abundance = stub_prep_params(pros=["ABC", "CCE", "AAB"], pro_abundances=[2, np.nan, 1])

            result = PrepResult.stub_prep_result(
                pros=[".", "ABCDEFGHI", "DDD"],
                pro_is_decoys=[False, False, True],
                pro_ptm_locs=["", "2;4"],
                peps=[".", "AAA", "CDE", "DDD"],
                pep_pro_iz=[0, 1, 1, 2],
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
                assert np.all([not np.isnan(x) for x in pros_abundance])

            zest()

        def it_gets_pros__ptm_locs():
            with_ptm_locs = result.pros__ptm_locs()
            assert len(with_ptm_locs) == 1
            assert np.all(with_ptm_locs.pro_ptm_locs != "")
            assert np.all(with_ptm_locs.columns == PrepResult.pros_columns)

        def it_gets_pros__from_decoys():
            from_decoys = result.pros__from_decoys()
            assert len(from_decoys) == 1
            assert np.all(from_decoys.pro_is_decoy)
            assert np.all(from_decoys.columns == PrepResult.pros_columns)

        def it_gets_pros__no_decoys():
            no_decoys = result.pros__no_decoys()
            assert len(no_decoys) == 2
            assert not np.any(no_decoys.pro_is_decoy)
            assert np.all(no_decoys.columns == PrepResult.pros_columns)

        def it_gets_proseqs():
            proseqs = result.proseqs()
            assert proseqs.query('pro_i == 1').aa.str.cat() == 'ABCDEFGHI'
            assert np.all(proseqs.columns == PrepResult.pro_seqs_columns)

        def it_gets_prostrs():
            prostrs = result.prostrs()
            assert len(prostrs) == result.n_pros
            assert "seqstr" in prostrs.columns
            assert prostrs.loc[1, "seqstr"] == 'ABCDEFGHI'

        zest()

    def pros_of_interest():
        result = None

        def _before():
            nonlocal result
            result = PrepResult.stub_prep_result(
                pros=["ABCD", "AACB"],
                pro_is_decoys=[False, False],
                peps=["AA", "AB"],
                pep_pro_iz=[1,0]
            )
            result.params = stub_prep_params(["ABCD"])

        def it_sets_pros_of_interest():
            result.set_pros_of_interest('id_0')
            assert result.pros().set_index('pro_id').at['id_0', 'pro_report'] > 0

        def it_gets_n_pros_of_interest():
            assert result.n_pros_of_interest == 0
            result.set_pros_of_interest(['id_0', 'id_1'])
            assert result.n_pros_of_interest == 2

        def it_asserts_protein_ids():
            with zest.raises(Exception):
                result.set_pros_of_interest('P1')

        def it_sets_pro_ptm_locs():
            result.set_pro_ptm_locs('id_0', '1;3')
            assert result.pros().set_index('pro_id').at['id_0', 'pro_ptm_locs'] == '1;3'
            assert 'id_0' in result.pros__ptm_locs().pro_id.values

        def it_gets_pro_ptm_locs():
            assert result.get_pro_ptm_locs('id_0') == ''
            result.set_pro_ptm_locs('id_0', '1;3')
            assert result.get_pro_ptm_locs('id_0') == '1;3'

        zest()

    @zest.skip("m", "Manas")
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

    @zest.skip("m", "Manas")
    def pros_and_peps():
        def it_gets_pros__peps():
            raise NotImplementedError

        def it_gets_pros__peps__pepstrs():
            raise NotImplementedError

        zest()

    zest()
