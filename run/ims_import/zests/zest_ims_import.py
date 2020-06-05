from munch import Munch
from plumbum import local
import tempfile
import numpy as np
from zest import zest
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import import ims_import_worker as worker
from plaster.tools.log.log import debug


class ND2(Munch):
    def get_fields(self):
        ims = np.full((3, self.n_channels, self.dim[0], self.dim[1]), 0.0)
        if self._fill_by == "channel":
            for ch in range(self.n_channels):
                ims[:, ch, :, :] = ch
        elif self._fill_by == "cycle":
            for cy in range(self._n_cycles):
                ims[cy, :, :, :] = cy
        return ims


@zest.group("integration")
def zest_ims_import():
    tmp_src = tempfile.NamedTemporaryFile()
    tmp_dst = tempfile.TemporaryDirectory()

    src_path = local.path(tmp_src.name)
    with local.cwd(local.path(tmp_dst.name)):

        m_scan_nd2_files = zest.stack_mock(worker._scan_nd2_files)
        m_scan_tif_files = zest.stack_mock(worker._scan_tif_files)
        m_scan_npy_files = zest.stack_mock(worker._scan_npy_files)
        m_nd2 = zest.stack_mock(worker._nd2)

        n_cycles = 2
        n_fields = 3
        n_channels = 4
        cycle_files = [(src_path / f"{i}.nd2") for i in range(n_cycles)]

        def _make_nd2(dim, fill_by="channel", n_cycles=None):
            return ND2(
                n_fields=n_fields,
                n_channels=n_channels,
                dim=(dim, dim),
                x=[0] * n_fields,
                y=[0] * n_fields,
                z=[0] * n_fields,
                pfs_status=[0] * n_fields,
                pfs_offset=[0] * n_fields,
                exposure_time=[0] * n_fields,
                camera_temp=[0] * n_fields,
                _fill_by=fill_by,
                _n_cycles=n_cycles,
            )

        ims_import_params = None
        nd2 = None

        def _before():
            nonlocal ims_import_params, nd2
            ims_import_params = ImsImportParams()
            nd2 = _make_nd2(64)
            m_nd2.returns(nd2)
            m_scan_nd2_files.returns(cycle_files)
            m_scan_tif_files.returns([])
            m_scan_npy_files.returns([])

        def it_scatter_gathers():
            result = worker.ims_import(src_path, ims_import_params)
            emitted_files = list(local.path(".").walk())
            assert len(emitted_files) == 9
            assert result.params == ims_import_params
            assert result.n_fields == n_fields
            assert result.n_channels == n_channels
            assert result.n_cycles == n_cycles

        def it_converts_to_power_of_2():
            with zest.mock(worker._convert_message):
                nd2 = _make_nd2(63)
                m_nd2.returns(nd2)
                result = worker.ims_import(src_path, ims_import_params)
                assert result.field_chcy_ims(0).shape == (n_channels, n_cycles, 64, 64)

        def it_limits_fields():
            ims_import_params.n_fields_limit = 1
            result = worker.ims_import(src_path, ims_import_params)
            assert result.n_fields == 1

        def it_imports_src_channels():
            result = worker.ims_import(src_path, ims_import_params)
            assert np.all(result.field_chcy_ims(0)[0, :, :, :] == 0.0)
            assert np.all(result.field_chcy_ims(0)[1, :, :, :] == 1.0)

        def it_can_skip_fields():
            ims_import_params.start_field = 1
            result = worker.ims_import(src_path, ims_import_params)
            assert result.n_fields == 2

        def it_can_limit_cycles():
            ims_import_params.n_cycles_limit = 1
            result = worker.ims_import(src_path, ims_import_params)
            assert result.n_fields == n_fields
            assert result.n_cycles == n_cycles - 1
            assert result.n_channels == n_channels

        def it_can_skip_cycles():
            ims_import_params.start_cycle = 1
            result = worker.ims_import(src_path, ims_import_params)
            assert result.n_fields == n_fields
            assert result.n_cycles == n_cycles - 1
            assert result.n_channels == n_channels

        def movies():
            def _before():
                nonlocal ims_import_params, nd2
                ims_import_params = ImsImportParams(is_movie=True)
                nd2 = _make_nd2(64, "cycle", n_fields)
                m_nd2.returns(nd2)
                m_scan_nd2_files.returns(cycle_files)
                m_scan_tif_files.returns([])

            def it_swaps_fields_cycles():
                result = worker.ims_import(src_path, ims_import_params)
                assert result.n_cycles == n_fields
                assert result.n_fields == n_cycles
                assert result.n_channels == n_channels
                for cy in range(result.n_cycles):
                    assert np.all(result.field_chcy_ims(0)[:, cy, :, :] == float(cy))

            def it_can_limit_cycles():
                ims_import_params.n_cycles_limit = 2
                result = worker.ims_import(src_path, ims_import_params)
                assert result.n_cycles == 2
                assert result.n_fields == n_cycles
                assert result.n_channels == n_channels
                for cy in range(result.n_cycles):
                    assert np.all(result.field_chcy_ims(0)[:, cy, :, :] == float(cy))

            def it_can_skip_cycles():
                ims_import_params.start_cycle = 1
                result = worker.ims_import(src_path, ims_import_params)
                assert result.n_cycles == n_fields - 1
                assert result.n_fields == n_cycles
                assert result.n_channels == n_channels
                for cy in range(result.n_cycles):
                    assert np.all(
                        result.field_chcy_ims(0)[:, cy, :, :] == float(cy + 1)
                    )

            def it_converts_to_power_of_2():
                with zest.mock(worker._convert_message):
                    nd2 = _make_nd2(63, "cycle", n_fields)
                    m_nd2.returns(nd2)
                    result = worker.ims_import(src_path, ims_import_params)
                    assert result.field_chcy_ims(0).shape == (
                        n_channels,
                        n_fields,
                        64,
                        64,
                    )

            zest()

        zest()


def zest_ims_import_from_npy():
    tmp_dst = tempfile.TemporaryDirectory()
    with local.cwd(local.path(tmp_dst.name)):
        m_scan_nd2_files = zest.stack_mock(worker._scan_nd2_files)
        m_scan_tif_files = zest.stack_mock(worker._scan_tif_files)
        m_scan_npy_files = zest.stack_mock(worker._scan_npy_files)
        m_load_npy = zest.stack_mock(worker._load_npy)

        npy_files = [
            # area, field, channel, cycle
            "area_000_cell_000_555nm_001.npy",
            "area_000_cell_000_647nm_001.npy",
            "area_000_cell_000_555nm_002.npy",
            "area_000_cell_000_647nm_002.npy",
            "area_000_cell_000_555nm_003.npy",
            "area_000_cell_000_647nm_003.npy",
            "area_000_cell_001_555nm_001.npy",
            "area_000_cell_001_647nm_001.npy",
            "area_000_cell_001_555nm_002.npy",
            "area_000_cell_001_647nm_002.npy",
            "area_000_cell_001_555nm_003.npy",
            "area_000_cell_001_647nm_003.npy",
        ]

        ims_import_params = None

        def _before():
            nonlocal ims_import_params
            ims_import_params = ImsImportParams()
            m_scan_nd2_files.returns([])
            m_scan_tif_files.returns([])
            m_scan_npy_files.returns(npy_files)
            m_load_npy.returns(np.zeros((16, 16)))

        def it_scans_npy_arrays():
            (
                mode,
                nd2_paths,
                tif_paths_by_field_channel_cycle,
                npy_paths_by_field_channel_cycle,
                n_fields,
                n_channels,
                n_cycles,
                dim,
            ) = worker._scan_files("")

            assert mode == "npy"
            assert nd2_paths == []
            assert tif_paths_by_field_channel_cycle == {}
            assert (
                local.path(npy_paths_by_field_channel_cycle[(0, 0, 0)]).name
                == npy_files[0]
            )
            assert n_fields == 2 and n_channels == 2 and n_cycles == 3
            assert dim == (16, 16)

        def it_ims_import_npy():
            res = worker.ims_import(
                ".", ims_import_params, progress=None, pipeline=None
            )
            assert res.n_fields == 2 and res.n_channels == 2 and res.n_cycles == 3

        zest()
