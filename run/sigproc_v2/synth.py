from plumbum import local
import numpy as np
from plaster.tools.utils import utils
from plaster.tools.image import imops
from plaster.tools.image.coord import XY, YX, WH, HW, ROI
from plaster.tools.log.log import debug, important
from plaster.run.sigproc_v2.psf_sample import psf_sample


class Synth:
    """
    Generate synthetic images for testing.

    This system is organized so that synthetic image(s) is
    delayed until the render() command is called. This allows
    for "reaching in" to the state and messing with it.

    Example, suppose that in some strange test you need to
    have a position of a certain peak location in very specific
    places for the test. To prevent a proliferation of one-off
    methods in this class, the idea is that you can use the
    method that creates two peaks and then "reach in" to
    tweak the positions directly before render.

    Examples:
        with Synth() as s:
            p = PeaksModelGaussian()
            p.locs_randomize()
            CameraModel(100, 2)
            s.render()

    """

    synth = None

    def __init__(
        self,
        n_fields=1,
        n_channels=1,
        n_cycles=1,
        dim=(512, 512),
        save_as=None,
        overwrite=False,
    ):
        self.n_fields = n_fields
        self.n_channels = n_channels
        self.n_cycles = n_cycles
        self.dim = dim
        self.save_as = save_as
        self.models = []
        self.aln_offsets = np.random.uniform(-20, 20, size=(self.n_cycles, 2)).astype(
            int
        )
        self.aln_offsets[0] = (0, 0)
        if not overwrite:
            assert Synth.synth is None
        Synth.synth = self

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        Synth.synth = None
        if exception_type is not None:
            raise exception_type(exception_value)
        ims = self.render_flchcy()
        self._save_np(ims, "ims")

    def add_model(self, model):
        self.models += [model]

    def _save_np(self, arr, name):
        if self.save_as is not None:
            save_as = local.path(self.save_as) + f"_{name}"
            np.save(save_as, arr)
            important(f"Wrote synth image to {save_as}.npy")

    def render_chcy(self):
        """
        Returns only chcy_ims (first field)
        """
        ims = np.zeros((self.n_channels, self.n_cycles, *self.dim))
        for ch_i in np.arange(self.n_channels):
            for cy_i in np.arange(self.n_cycles):
                im = ims[ch_i, cy_i]
                for model in self.models:
                    model.render(im, cy_i)

                ims[ch_i, cy_i] = imops.shift(im, self.aln_offsets[cy_i])

        return ims

    def render_flchcy(self):
        flchcy_ims = np.zeros(
            (self.n_fields, self.n_channels, self.n_cycles, *self.dim)
        )
        for fl_i in range(self.n_fields):
            flchcy_ims[fl_i] = self.render_chcy()
        return flchcy_ims

    def scale_peaks_by_max(self):
        """
        For some tests it is nice to know that the max brightness of a peak
        instead of the area under the curve.
        """
        self.peak_ims = [peak_im / np.max(peak_im) for peak_im in self.peak_ims]


class BaseSynthModel:
    def __init__(self):
        self.dim = Synth.synth.dim
        Synth.synth.add_model(self)

    def render(self, im, cy_i):
        pass


class PeaksModel(BaseSynthModel):
    def __init__(self, n_peaks=1000):
        super().__init__()
        self.n_peaks = n_peaks
        self.locs = np.zeros((n_peaks, 2))
        self.amps = np.ones((n_peaks,))

    def locs_randomize(self):
        self.locs = np.random.uniform(0, self.dim, (self.n_peaks, 2))
        return self

    def locs_randomize_away_from_edges(self):
        self.locs = np.random.uniform(
            [15, 15], np.array(self.dim) - 15, (self.n_peaks, 2)
        )
        return self

    def locs_grid(self, steps=10):
        pad = 10
        self.locs = [
            (y, x)
            for y in utils.ispace(pad, self.dim[0] - 2 * pad, steps)
            for x in utils.ispace(pad, self.dim[0] - 2 * pad, steps)
        ]
        return self

    def amps_constant(self, val):
        self.amps = val * np.ones((self.n_peaks,))
        return self

    def amps_randomize(self, mean=1000, std=10):
        self.amps = mean + std * np.random.randn(self.n_peaks)
        return self

    def remove_near_edges(self, dist=20):
        self.locs = np.array(
            [
                loc
                for loc in self.locs
                if loc[0] > dist
                and loc[0] < self.dim[0] - dist
                and loc[1] > dist
                and loc[1] < self.dim[1] - dist
            ]
        )
        return self


class PeaksModelGaussian(PeaksModel):
    def __init__(self, **kws):
        super().__init__(**kws)
        self.std = None
        self.std_x = None
        self.std_y = None

    def render(self, im, cy_i):
        if self.std_x is None:
            self.std_x = [self.std]
        if self.std_y is None:
            self.std_y = [self.std]

        n_locs = len(self.locs)
        if len(self.std_x) != n_locs:
            self.std_x = np.repeat(self.std_x, (n_locs,))
        if len(self.std_y) != n_locs:
            self.std_y = np.repeat(self.std_y, (n_locs,))

        super().render(im, cy_i)
        mea = 17
        for loc, amp, std_x, std_y in zip(self.locs, self.amps, self.std_x, self.std_y):
            frac_x = np.modf(loc[0])[0]
            frac_y = np.modf(loc[1])[0]
            peak_im = imops.gauss2_rho_form(
                amp=amp,
                std_x=std_x,
                std_y=std_y,
                pos_x=mea // 2 + frac_x,
                pos_y=mea // 2 + frac_y,
                rho=0.0,
                const=0.0,
                mea=mea,
            )

            imops.accum_inplace(im, peak_im, loc=YX(*np.floor(loc)), center=True)


class PeaksModelGaussianCircular(PeaksModelGaussian):
    def __init__(self, **kws):
        super().__init__(**kws)
        self.std = 1.0

    def widths_uniform(self, std=1.5):
        self.std = std
        return self

    def render(self, im, cy_i):
        # self.covs = np.array([(std ** 2) * np.eye(2) for std in self.stds])
        super().render(im, cy_i)


class PeaksModelGaussianAstigmatism(PeaksModelGaussian):
    def __init__(self, strength, **kws):
        raise DeprecationWarning
        super().__init__(**kws)
        self.strength = strength
        center = np.array(self.dim) / 2
        d = self.dim[0]
        for loc_i, pos in enumerate(self.locs):
            delta = center - pos
            a = np.sqrt(np.sum(delta ** 2))
            r = 1 + strength * a / d
            pc0 = delta / np.sqrt(delta.dot(delta))
            pc1 = np.array([-pc0[1], pc0[0]])
            cov = np.eye(2)
            cov[0, 0] = r * pc0[1]
            cov[1, 0] = r * pc0[0]
            cov[0, 1] = pc1[1]
            cov[1, 1] = pc1[0]
            self.covs[loc_i, :, :] = cov


class PeaksModelPSF(PeaksModel):
    def __init__(self, n_z_slices=8, depth_in_microns=0.4, r_in_microns=28.0, **kws):
        """
        Generates a set of psf images for each z slice called self.z_to_psf
        The self.z_iz keeps track of which z slice each peak is assigned to.
        """
        super().__init__(**kws)
        self.n_z_slices = n_z_slices
        self.z_iz = np.zeros((self.n_peaks,), dtype=int)
        self.z_to_psf = psf_sample(
            n_z_slices=64, depth_in_microns=depth_in_microns, r_in_microns=r_in_microns
        )

    def z_randomize(self):
        # Unrealisitically pull from any PSF z depth
        self.z_iz = np.random.randint(0, self.n_z_slices, self.n_peaks)
        return self

    def z_set_all(self, z_i):
        self.z_iz = (z_i * np.ones((self.n_peaks,))).astype(int)
        return self

    def render(self, im, cy_i):
        super().render(im, cy_i)
        for loc, amp, z_i in zip(self.locs, self.amps, self.z_iz):
            frac_part, int_part = np.modf(loc)
            shifted_peak_im = imops.sub_pixel_shift(self.z_to_psf[z_i], frac_part)
            imops.accum_inplace(
                im, amp * shifted_peak_im, loc=YX(*int_part), center=True
            )


class IlluminationQuadraticFalloffModel(BaseSynthModel):
    def __init__(self, center=(0.5, 0.5), width=1.2):
        super().__init__()
        self.center = center
        self.width = width

    def render(self, im, cy_i):
        super().render(im, cy_i)
        yy, xx = np.meshgrid(
            (np.linspace(0, 1, im.shape[0]) - self.center[0]) / self.width,
            (np.linspace(0, 1, im.shape[1]) - self.center[1]) / self.width,
        )
        self.regional_scale = np.exp(-(xx ** 2 + yy ** 2))
        im *= self.regional_scale


class CameraModel(BaseSynthModel):
    def __init__(self, bias=100, std=10):
        super().__init__()
        self.bias = bias
        self.std = std

    def render(self, im, cy_i):
        super().render(im, cy_i)
        bg = np.random.normal(loc=self.bias, scale=self.std, size=self.dim)
        imops.accum_inplace(im, bg, XY(0, 0), center=False)
