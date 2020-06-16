from zest import zest
import itertools
import numpy as np
import pandas as pd
from plaster.run.prep.prep_result import PrepResult
from plaster.run.prep.prep_params import PrepParams


def _stub_prep_params(pros, pro_abundances=[]):
    def stub_protein(i, seq, abundance):
        pro = dict(name=f"id_{i}", sequence=seq)
        if abundance is not None:
            pro["abundance"] = abundance
        return pro

    proteins = [
        stub_protein(i, seq, abundance)
        for i, (seq, abundance) in enumerate(
            itertools.zip_longest(pros, pro_abundances)
        )
    ]

    return PrepParams(proteins=proteins)


def zest_PrepResult():
    def pros():
        result = None
        default_params = None
        params_with_abundance = None

        def _before():
            nonlocal result, default_params, params_with_abundance

            default_params = _stub_prep_params(pros=["ABC", "CCE", "AAB"])
            params_with_abundance = _stub_prep_params(
                pros=["ABC", "CCE", "AAB"], pro_abundances=[2, np.nan, 1]
            )

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
            assert with_ptm_locs.columns.tolist() == PrepResult.pros_columns

        def it_gets_pros__from_decoys():
            from_decoys = result.pros__from_decoys()
            assert len(from_decoys) == 1
            assert np.all(from_decoys.pro_is_decoy)
            assert from_decoys.columns.tolist() == PrepResult.pros_columns

        def it_gets_pros__no_decoys():
            no_decoys = result.pros__no_decoys()
            assert len(no_decoys) == 2
            assert not np.any(no_decoys.pro_is_decoy)
            assert no_decoys.columns.tolist() == PrepResult.pros_columns

        def it_gets_proseqs():
            proseqs = result.proseqs()
            assert proseqs.query("pro_i == 1").aa.str.cat() == "ABCDEFGHI"
            assert proseqs.columns.tolist() == PrepResult.pro_seqs_columns

        def it_gets_prostrs():
            prostrs = result.prostrs()
            assert len(prostrs) == result.n_pros
            assert "seqstr" in prostrs.columns
            assert prostrs.loc[1, "seqstr"] == "ABCDEFGHI"

        zest()

    def pros_of_interest():
        result = None

        def _before():
            nonlocal result
            result = PrepResult.stub_prep_result(
                pros=["ABCD", "AACB"],
                pro_is_decoys=[False, False],
                peps=["AA", "AB"],
                pep_pro_iz=[1, 0],
            )
            result.params = _stub_prep_params(["ABCD"])

        def it_sets_pros_of_interest():
            result.set_pros_of_interest("id_0")
            assert result.pros().pro_report.astype(bool).tolist() == [True, False]
            result.set_pros_of_interest("id_1")
            assert result.pros().pro_report.astype(bool).tolist() == [False, True]
            result.set_pros_of_interest(["id_0", "id_1"])
            assert result.pros().pro_report.astype(bool).tolist() == [True, True]

        def it_gets_pros__in_report():
            result.set_pros_of_interest("id_1")
            in_report = result.pros__in_report()
            assert len(in_report) == 1
            assert in_report.set_index("pro_id").index.tolist() == ["id_1"]

        def it_gets_n_pros_of_interest():
            assert result.n_pros_of_interest == 0
            result.set_pros_of_interest(["id_0", "id_1"])
            assert result.n_pros_of_interest == 2

        def it_asserts_protein_ids():
            with zest.raises(Exception):
                result.set_pros_of_interest("P1")

        def it_sets_pro_ptm_locs():
            result.set_pro_ptm_locs("id_0", "1;3")
            assert result.pros().set_index("pro_id").at["id_0", "pro_ptm_locs"] == "1;3"
            assert "id_0" in result.pros__ptm_locs().pro_id.values

        def it_gets_pro_ptm_locs():
            assert result.get_pro_ptm_locs("id_0") == ""
            result.set_pro_ptm_locs("id_0", "1;3")
            assert result.get_pro_ptm_locs("id_0") == "1;3"

        zest()

    def peps():
        result = None
        all_decoys_result = None

        def _before():
            nonlocal result, all_decoys_result

            result = PrepResult.stub_prep_result(
                pros=["ABCD", "AACB"],
                pro_is_decoys=[False, False],
                peps=["AACB", "ABCD"],
                pep_pro_iz=[1, 0],
            )
            result.params = _stub_prep_params(["AACB", "ABCD"], [5])

            all_decoys_result = PrepResult.stub_prep_result(
                pros=["ABC", "CDE"],
                pro_is_decoys=[True, True],
                peps=["ABC", "CDE"],
                pep_pro_iz=[0, 1],
            )

        def it_gets_peps():
            peps = result.peps()
            assert isinstance(peps, pd.DataFrame)
            assert peps.columns.tolist() == PrepResult.peps_columns
            assert len(peps) == 2

        def it_gets_n_peps():
            assert result.n_peps == 2

        def it_gets_pepstrs():
            pepstrs = result.pepstrs()
            assert pepstrs.seqstr.values.tolist() == ["AACB", "ABCD"]

        def it_gets_pepseqs():
            pepseqs = result.pepseqs()
            assert pepseqs.columns.tolist() == PrepResult.pep_seqs_columns
            assert pepseqs.aa.str.cat() == "AACBABCD"
            assert len(pepseqs) == 8

        def it_gets_peps_abundance():
            peps_abundance = result.peps_abundance()
            assert len(peps_abundance) == 2

            def it_fills_zero_for_missing_abundance():
                assert peps_abundance.tolist() == [5.0, 0.0]

            zest()

        def it_gets_peps__no_decoys():
            no_decoys = result.peps__no_decoys()
            assert len(no_decoys) == 2
            assert no_decoys.columns.tolist() == PrepResult.peps_columns

            def it_handles_empty_return():
                no_decoys = all_decoys_result.peps__no_decoys()
                assert len(no_decoys) == 0
                assert no_decoys.columns.tolist() == PrepResult.peps_columns

            zest()

        def it_gets_peps__from_decoys():
            from_decoys = all_decoys_result.peps__from_decoys()
            assert len(from_decoys) == 2
            assert from_decoys.columns.tolist() == PrepResult.peps_columns

            def it_handles_empty_return():
                from_decoys = result.peps__from_decoys()
                assert len(from_decoys) == 0
                assert from_decoys.columns.tolist() == PrepResult.peps_columns

            zest()

        def peps__in_report():
            result.set_pros_of_interest("id_0")
            in_report = result.peps__in_report()
            assert in_report.columns.tolist() == PrepResult.peps_columns
            assert len(in_report) == 1
            assert in_report.set_index("pro_i").index.tolist() == [0]

            def it_handles_empty_return():
                no_interest_result = PrepResult.stub_prep_result(
                    pros=["ABC", "CDE"],
                    pro_is_decoys=[False, False],
                    peps=["ABC", "CDE"],
                    pep_pro_iz=[0, 1],
                )
                no_interest_result.params = _stub_prep_params(["ABC", "CDE"])
                in_report = no_interest_result.peps__in_report()
                assert len(in_report) == 0

            zest()

        def it_gets_peps__pepseqs():
            pepseqs = result.peps__pepseqs()
            assert len(pepseqs) == 8
            assert pepseqs.aa.str.cat() == "AACBABCD"

        def it_gets_pepseqs__no_decoys():
            result = PrepResult.stub_prep_result(
                pros=["ABCD", "AACB"],
                pro_is_decoys=[True, False],
                peps=["ABCD", "AACB"],
                pep_pro_iz=[0, 1],
            )
            result.params = _stub_prep_params(["ABCD", "AACB"])
            pepseqs = result.pepseqs__no_decoys()
            assert len(pepseqs) == 4
            assert pepseqs.aa.str.cat() == "AACB"

        def peps__ptms():
            result = PrepResult.stub_prep_result(
                pros=[".", "ABCDEF", "ABACAB"],
                pro_is_decoys=[False, True, False],
                pro_ptm_locs=["", "3;4", "1;3"],
                peps=[".", "ABCDEF", "ABACAB"],
                pep_pro_iz=[0, 1, 2],
            )
            result._peps.loc[1, "pep_start"] = 1
            result._peps.loc[1, "pep_stop"] = 2
            result._peps.loc[2, "pep_start"] = 0
            result._peps.loc[2, "pep_stop"] = 6
            result.params = _stub_prep_params([".", "ABCDEF", "ABACAB"])

            peps__ptms = result.peps__ptms(ptms_to_rows=False)
            assert len(peps__ptms) == 1
            assert peps__ptms.at[0, "pro_id"] == "id_2"
            assert peps__ptms.at[0, "n_pep_ptms"] == 2
            assert peps__ptms.at[0, "pro_ptm_locs"] == "1;3"

            def it_filters_decoys():
                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=False,
                    ptm_peps_only=False,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 2
                assert peps__ptms["pro_id"].tolist() == ["id_1", "id_2"]

                peps__ptms = result.peps__ptms(
                    include_decoys=False,
                    in_report_only=False,
                    ptm_peps_only=False,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 1
                assert peps__ptms["pro_id"].tolist() == ["id_2"]

            def it_filters_in_report_only():
                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=True,
                    ptm_peps_only=False,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 0

                result.set_pros_of_interest("id_1")
                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=True,
                    ptm_peps_only=False,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 1
                assert peps__ptms["pro_id"].tolist() == ["id_1"]

            def it_filters_ptm_peps_only():
                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=False,
                    ptm_peps_only=False,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 2
                assert peps__ptms["pro_id"].tolist() == ["id_1", "id_2"]
                assert peps__ptms["pro_ptm_locs"].tolist() == ["", "1;3"]

                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=False,
                    ptm_peps_only=True,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 1
                assert peps__ptms["pro_id"].tolist() == ["id_2"]
                assert peps__ptms["pro_ptm_locs"].tolist() == ["1;3"]

            def it_ptms_to_rows():
                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=False,
                    ptm_peps_only=True,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 1
                assert peps__ptms["pro_id"].tolist() == ["id_2"]
                assert peps__ptms["pro_ptm_locs"].tolist() == ["1;3"]

                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=False,
                    ptm_peps_only=True,
                    ptms_to_rows=True,
                )
                assert len(peps__ptms) == 2
                assert peps__ptms["pro_id"].tolist() == ["id_2", "id_2"]
                assert peps__ptms["pro_ptm_locs"].tolist() == ["1;3", "1;3"]
                assert peps__ptms["ptm"].tolist() == ["1", "3"]

            zest()

        zest()

    def pros_and_peps():
        result = PrepResult.stub_prep_result(
            pros=[".", "ABCDEF", "ABACAB"],
            pro_is_decoys=[False, True, False],
            peps=[".", "ABCDEF", "ABACAB"],
            pep_pro_iz=[0, 1, 2],
        )
        result.params = _stub_prep_params([".", "ABCDEF", "ABACAB"])

        def it_gets_pros__peps():
            pros__peps = result.pros__peps()
            assert len(pros__peps) == 3
            assert pros__peps["pro_id"].tolist() == ["id_0", "id_1", "id_2"]
            assert pros__peps["pro_i"].tolist() == [0, 1, 2]
            assert pros__peps["pep_i"].tolist() == [0, 1, 2]
            assert pros__peps["pro_is_decoy"].tolist() == [False, True, False]

        def it_gets_pros__peps__pepstrs():
            pros__peps__pepstrs = result.pros__peps__pepstrs()
            assert len(pros__peps__pepstrs) == 3
            assert pros__peps__pepstrs["pro_id"].tolist() == ["id_0", "id_1", "id_2"]
            assert pros__peps__pepstrs["seqstr"].tolist() == [".", "ABCDEF", "ABACAB"]

        zest()

    zest()
