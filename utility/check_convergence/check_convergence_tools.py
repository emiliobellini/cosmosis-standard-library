import argparse
import configparser
import json
import matplotlib.pyplot as plt
import numpy as np
import os


# ------------------- Parser -------------------------------------------------#

def argument_parser():
    """ Call the parser to read command line arguments.

    Args:
        None.

    Returns:
        args: the arguments read by the parser

    """

    parser = argparse.ArgumentParser(
        'Test convergence computing the autocorrelation of'
        'parameters. It works only for emcee.')

    # Arguments
    parser.add_argument('params_ini', type=str,
                        help='Parameters file (ini)')
    parser.add_argument('chain_file', type=str,
                        help='Chain file')
    parser.add_argument('--plots', '-p', action='store_true',
                        help='Save plots.')
    parser.add_argument('--num_auto_points', '-n', type=int, default=1,
                        help='Number of autocorrelation points to be computed')

    return parser.parse_args()


# ------------------- IniFile ------------------------------------------------#

class IniFile(object):
    def __init__(self, path):
        self.top_section_ini = 'top_section'  # Name of the top section
        if not os.path.isfile(path):
            raise IOError('Ini file not found at {}'.format(path))
        self.path = os.path.abspath(path)
        self.content = None
        self.sections = []
        self.keys = []
        return

    def __setitem__(self, item, value):
        self.content[item] = value

    def __getitem__(self, item):
        return self.content[item]

    def read(self):
        """
        Read the ini file and store the content. It manually creates a
        top section to store all the content at the beginning of the file
        (this allows to have a ini file without sections).
        """
        config = configparser.ConfigParser(
            inline_comment_prefixes=('#', ';'),
            empty_lines_in_values=False
        )
        config.optionxform = str
        with open(self.path) as fn:
            u = '[{}]\n'.format(self.top_section_ini) + fn.read()
            try:
                config.read_string(u)
            except TypeError:  # Added for compatibility with Python2
                config.read_string(unicode(u))  # noqa: F821
        self.content = json.loads(json.dumps(config._sections))

        # Get sections and keys
        for sec in self.content:
            if self.content[sec]:
                self.sections.append(sec)
                for key in self.content[sec]:
                    self.keys.append((sec, key))

        return self


# ------------------- Autocorrelator -----------------------------------------#

class ChainFile(object):
    def __init__(self, path):
        if not os.path.isfile(path):
            raise IOError('Chain file not found at {}'.format(path))
        self.path = os.path.abspath(path)
        self.content = None
        self.parameters = []
        return

    def __setitem__(self, item, value):
        self.content[item] = value

    def __getitem__(self, item):
        return self.content[item]

    def reshape(self, n_walkers):
        # Input chain with dimensions
        # (n_points*n_walkers, n_parameters)/
        # Return the chain as an array with
        # dimensions: (n_points, n_walkers, n_parameters)
        if self.content is None:
            self.read()
        chain = self.content
        n_parameters = chain.shape[1]
        self.content = chain.reshape((-1, n_walkers, n_parameters))
        self.shape = self.content.shape
        return self

    def read(self):
        with open(self.path) as fn:
            names = fn.readline()
        names = names.strip('#').strip('\n').split('\t')
        self.names = [tuple(name.split('--')) for name in names]
        # Read array
        self.content = np.genfromtxt(self.path)
        self.shape = self.content.shape
        return self

    def _next_pow_two(self, n):
        i = 1
        while i < n:
            i = i << 1
        return i

    def _autocorr_func_1d(self, x, norm=True):
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError('Invalid dimensions for 1D autocorrelation'
                             ' function')
        n = self._next_pow_two(len(x))

        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
        acf /= 4 * n

        # Optionally normalize
        if norm:
            acf /= acf[0]

        return acf

    def _auto_window(self, taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    def autocorr(self, y, c=5.0):
        f = np.zeros(y.shape[1])
        for yy in y:
            f += self._autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = self._auto_window(taus, c)
        return taus[window]

    def estimate_autocorrelations(self, plots=False, num=1):
        if plots:
            plots_path = os.path.splitext(self.path)[0] + '_plots'
            try:
                os.mkdir(plots_path)
            except FileExistsError:
                pass
        print('Computing autocorrelations:')
        if num <= 1:
            len_chains = [self.shape[0]]
        else:
            len_chains = np.exp(np.linspace(
                np.log(100), np.log(self.shape[0]), num)).astype(int)
        for npar, par in enumerate(self.names):
            if len(par) == 2:
                auto = [self.autocorr(self[:n, :, npar]) for n in len_chains]
                print('----> Parameter: {}, {}. N_steps/100 = '
                      '{:.2f} should be larger than tau = {:.2f}'
                      ''.format(par[0], par[1], len_chains[-1]/100, auto[-1]))
                if plots:
                    plt.loglog(len_chains, auto,
                               'o-', label=r"$\tau_{\rm estimate}$")
                    plt.plot(len_chains, len_chains/100.0,
                             '--b', label=r"$\tau = N/100$")
                    plt.plot(len_chains, len_chains/1000.0,
                             '--r', label=r"$\tau = N/1000$")
                    plt.xlabel("number of samples, $N$")
                    plt.ylabel(r"$\tau$")
                    plt.legend(fontsize=14)
                    plt.title(r"{}, {}".format(par[0], par[1]))
                    fname = 'tau_{}_{}.pdf'.format(par[0], par[1])
                    plt.savefig(os.path.join(plots_path, fname))
                    plt.close
        return
