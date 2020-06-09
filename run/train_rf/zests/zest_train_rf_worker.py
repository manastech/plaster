import numpy as np
from zest import zest
from plaster.run.train_rf import train_rf_worker as worker


def zest_train_rf():
    def it_subsamples():
        n_samples = 8
        n_peptides = 5
        n_cycles = 3

        y0 = np.repeat(np.arange(n_peptides), (n_samples,))
        X0 = np.random.uniform(size=(n_samples * n_peptides, n_cycles))

        n_subsample = 2
        X1, y1 = worker._subsample(n_subsample, X0, y0)
        assert np.all( y1 == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] )

        assert X1.shape == (n_subsample * n_peptides, n_cycles)

        assert np.all(np.isin(X1[0, :], X0[0:n_samples]))
        assert np.all(np.isin(X1[1, :], X0[0:n_samples]))

        assert np.all(np.isin(X1[8, :], X0[(n_samples-1)*n_peptides:(n_samples+0)*n_peptides]))
        assert np.all(np.isin(X1[9, :], X0[(n_samples-1)*n_peptides:(n_samples+0)*n_peptides]))

    zest()
