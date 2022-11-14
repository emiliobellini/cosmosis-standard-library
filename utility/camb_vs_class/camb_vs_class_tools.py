import argparse
import configparser
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess as sp
from scipy import interpolate
from tabulate import tabulate


# ------------------- Parser -------------------------------------------------#

def argument_parser():
    """ Call the parser to read command line arguments.

    Args:
        None.

    Returns:
        args: the arguments read by the parser

    """

    parser = argparse.ArgumentParser('Run Camb and (hi_)class within cosmosis '
                                     'to check their output is equivalent.')

    # Arguments
    parser.add_argument('params_ref', type=str,
                        help='Parameters file (ini) for the reference code')
    parser.add_argument('params_2nd', type=str,
                        help='Parameters file (ini) for the second code')
    parser.add_argument('--plots_path', '-p', type=str, help='Plots folder.')
    parser.add_argument('--save_plots', '-s', action='store_true', help='Save '
                        'plots without showing them (default: False).')
    parser.add_argument('--verbose', '-v', action='store_true',
                        default=False, help='Regulate verbosity.')

    return parser.parse_args()


# ------------------- Relative differences -----------------------------------#

def rel_diff1d(x1, y1, x2, y2, N=3e3, spacing='linear', epsilon=1e-5, ref=1):
    """ Relative difference between data vectors

    Args:
        x1,y1,x2,y2 (np array): data to interpolate.
        N (int): number of points on which to interpolate.
        spacing (str): how to populate the interpolation (log or linear).
        epsilon (float): cut the interval to avoid interpolation problems.
        ref (int): if 0 the rel diff is calculated using N points with linear
            or log spacing, if 1 (2) the rel diff is calculated only at the
            points of model 1 (2).

    Returns:
        (x,y): arrays with the relative difference.

    """

    # don't compute it many times, slow for long arrays
    xmin = (1.+0*epsilon)*max(min(x1), min(x2))
    xmax = (1.-0*epsilon)*min(max(x1), max(x2))

    data1 = interpolate.interp1d(x1, y1)
    data2 = interpolate.interp1d(x2, y2)

    if ref == 1:
        range = np.array([x for x in x1 if xmin <= x <= xmax])
    elif ref == 2:
        range = np.array([x for x in x2 if xmin <= x <= xmax])
    else:
        if spacing == 'linear':
            range = np.linspace(xmin, xmax, int(N))
        else:
            range = np.exp(np.linspace(np.log(xmin), np.log(xmax), int(N)))

    diff = np.array([data2(x)/data1(x)-1. for x in range])

    return range, diff


def rel_diff2d(x1, y1, z1, x2, y2, z2,
               N=3e2, spacing='linear', epsilon=1e-5, ref=1):
    """ Relative difference between data vectors

    Args:
        x1,y1,z1,x2,y2,z2 (np array): data to interpolate.
        N (int): number of points on which to interpolate.
        spacing (str): how to populate the interpolation (log or linear).
        epsilon (float): cut the interval to avoid interpolation problems.
        ref (int): if 0 the rel diff is calculated using N points with linear
            or log spacing, if 1 (2) the rel diff is calculated only at the
            points of model 1 (2).

    Returns:
        (x,y,z): arrays with the relative difference.

    """

    # don't compute it many times, slow for long arrays
    xmin = (1.+0*epsilon)*max(min(x1), min(x2))
    xmax = (1.-0*epsilon)*min(max(x1), max(x2))
    ymin = (1.+0*epsilon)*max(min(y1), min(y2))
    ymax = (1.-0*epsilon)*min(max(y1), max(y2))

    # consider providing z.T arrays directly as input
    data1 = interpolate.interp2d(x1, y1, z1.T)
    data2 = interpolate.interp2d(x2, y2, z2.T)

    if ref == 1:
        range_x = np.array([x for x in x1 if xmin <= x <= xmax])
        range_y = np.array([y for y in y1 if ymin <= y <= ymax])
    elif ref == 2:
        range_x = np.array([x for x in x2 if xmin <= x <= xmax])
        range_y = np.array([y for y in y2 if ymin <= y <= ymax])
    else:
        if spacing == 'linear':
            range_x = np.linspace(xmin, xmax, int(N))
            range_y = np.linspace(ymin, ymax, int(N))
        else:
            range_x = np.exp(np.linspace(np.log(xmin), np.log(xmax), int(N)))
            range_y = np.exp(np.linspace(np.log(ymin), np.log(ymax), int(N)))

    diff = np.zeros((range_x.shape[0], range_y.shape[0]))
    for nx, x in enumerate(range_x):
        diff[nx] = np.array([data2(x, y)/data1(x, y)-1. for y in range_y]).T

    return range_x, range_y, diff


# ------------------- Main Code class ----------------------------------------#

class Code(object):
    def __init__(self, ini_path=None):
        # Ini file
        self.top_section_ini = 'top_section'  # Name of the top section
        self.read_ini = False
        if ini_path:
            self._initialize_with_ini(ini_path)
        # Output
        self.read_output = False
        self.sections = []
        self.keys = {}
        self.data = {}
        # Settings
        self._indep_vars = ['ell', 'z', 'k_h']
        return None

    def _initialize_with_ini(self, path_ini):
        """
        Initialize Class attributes with ini_file
        """
        if not os.path.isfile(path_ini):
            raise IOError('Ini file not found at {}'.format(self.path_ini))
        self.path_ini = os.path.abspath(path_ini)
        self.content_ini = self._read_ini()
        self.name = self._get_name()
        self.path_data = self._get_path_data()
        self.path_plots = self._get_path_plots()
        self.read_ini = True
        return

    def _print_bf(self, text):
        str = '\033[1m{}\033[0m'.format(text)
        return str

    def _read_ini(self, path_ini=None):
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
        if path_ini:
            path = path_ini
        else:
            path = self.path_ini
        with open(path) as fn:
            u = '[{}]\n'.format(self.top_section_ini) + fn.read()
            try:
                config.read_string(u)
            except TypeError:  # Added for compatibility with Python2
                config.read_string(unicode(u))  # noqa: F821
        return json.loads(json.dumps(config._sections))

    def _get_name(self):
        """
        This gets the name of the code from the content of the ini
        file, as the last mudule run in the pipeline.
        """
        name = self.content_ini['pipeline']['modules']
        name = name.split(' ')[-1]
        name = name.strip()
        return name

    def _get_path_data(self):
        """
        This gets the data path from the content of the ini file.
        """
        path = self.content_ini['test']['save_dir']
        path = os.path.abspath(path)
        return path

    def _get_path_plots(self):
        """
        This gets the plot path.
        """
        path, _ = os.path.split(self.path_data)
        path = os.path.join(path, 'plots')
        path = os.path.abspath(path)
        return path

    def _path_exists_or_none(self, path):
        """ Check if a path exists, otherwise it returns error.

        Args:
            path: path to check.

        Returns:
            abspath: if the file exists it returns its absolute path

        """
        abspath = os.path.abspath(path)
        if os.path.exists(abspath):
            return abspath
        return None

    def _is_indep_var(self, keyname):
        if keyname in self._indep_vars:
            return True
        else:
            return False

    def _get_x_name(self, secname):
        set1 = set(self._indep_vars)
        set2 = set(self.data[secname].keys())
        inter = list(set1 & set2)
        if len(inter) == 1:
            return inter[0]
        else:
            raise IOError('Incorrect number of dimensions. Wanted 1, '
                          'got {}'.format(len(inter)))

    def _get_x_and_y_names(self, secname, zshape):
        set1 = set(self._indep_vars)
        set2 = set(self.data[secname].keys())
        inter = list(set1 & set2)
        if len(inter) == 2:
            if len(self.data[secname][inter[0]]) == zshape[0]:
                return inter[0], inter[1]
            else:
                return inter[1], inter[0]
        else:
            raise IOError('Incorrect number of dimensions. Wanted 2, '
                          'got {}'.format(len(inter)))

    def run(self, verbose=False, fake=False):
        """
        Run cosmosis using the ini file provided.
        This is meant to be run in test mode,
        but it can be generalised.
        """

        if fake:
            return

        if not self.read_ini:
            if self.path_ini:
                self._initialize_with_ini(self.path_ini)
            else:
                raise IOError('You must specify a valid .ini file if you want '
                              'to run the Code.')

        output = self._path_exists_or_none(self.path_data)
        if output:
            print('WARNING: output path {} already existent. '
                  'Removing files!'.format(self._print_bf(output)))
            shutil.rmtree(output)

        out = sp.run(['cosmosis', self.path_ini],
                     stdout=sp.PIPE, stderr=sp.PIPE, encoding='UTF-8')
        if verbose:
            self._print_bf('Stdout message:')
            print(out.stdout)
            self._print_bf('Stderr message:')
            print(out.stderr)

        return

    def load_data(self):
        """
        Load output data and store it in a nested dictionary.
        """
        path = self.path_data
        if not path:
            raise IOError('Path {} not found!'.format(path))

        for secname in os.listdir(path):
            secpath = os.path.join(path, secname)
            if os.path.isdir(secpath):
                self.sections.append(secname)
        self.sections.sort()

        for secname in self.sections:
            secpath = os.path.join(path, secname)
            self.keys[secname] = []
            for keyname in os.listdir(secpath):
                keypath = os.path.join(secpath, keyname)
                key = keyname.split('.')[0]
                if key == 'values':
                    continue
                else:
                    self.keys[secname].append(key)
        for secname in self.sections:
            self.keys[secname].sort()

        for secname in self.sections:
            if not self.keys[secname]:
                self.keys.pop(secname)
        self.sections = list(self.keys.keys())

        for secname in self.sections:
            self.data[secname] = {}
            for keyname in self.keys[secname]:
                keypath = os.path.join(path, secname, keyname+'.txt')
                self.data[secname][keyname] = \
                    np.genfromtxt(keypath, unpack=True)

        self.read_output = True
        return

    def print_keys(self):
        """
        Print nicely the sections and keys of the code.
        """

        if not self.read_output:
            raise IOError('You have to load the data first! (with load_data)')

        tot = []
        n_max_keys = max([len(self.keys[x]) for x in self.sections])
        for nsecname, secname in enumerate(self.sections):
            empty = (n_max_keys - len(self.keys[secname])) * ['']
            tot.append(np.array(self.keys[secname] + empty))
        print(self._print_bf('\n{} sections:'. format(self.name)))
        print(tabulate(np.array(tot).T,
              headers=self.sections, tablefmt="github"))
        return

    def diff_codes(self, code_2nd):
        """
        Return a Code class that contain the difference
        between two Codes.
        """
        tot_codes = [self] + [code_2nd]

        # Preliminary checks
        for code in tot_codes:
            if not code.name:
                code.name = 'unnamed_code'
            if not code.read_output:
                raise IOError('For Code {} you have to load the data first! '
                              '(with load_data)'.format(code.name))

        # Initialize new code
        joint = Code()
        name = '_vs_'.join([x.name for x in tot_codes])
        joint.name = name
        if self.path_plots:
            joint.path_plots = self.path_plots

        # Get common sections
        secs = [x.sections for x in tot_codes]
        joint.sections = list(set.intersection(*map(set, secs)))
        joint.sections.sort()

        # Get common keys
        for secname in joint.sections:
            keys = [x.keys[secname] for x in tot_codes]
            joint.keys[secname] = list(set.intersection(*map(set, keys)))
            joint.keys[secname].sort()

        # Get common data
        for secname in joint.sections:
            joint.data[secname] = {}
            for keyname in joint.keys[secname]:
                ndim = self.data[secname][keyname].ndim
                if not self._is_indep_var(keyname):
                    if ndim == 1:
                        xname = self._get_x_name(secname)
                        x1 = self.data[secname][xname]
                        y1 = self.data[secname][keyname]
                        x2 = code_2nd.data[secname][xname]
                        y2 = code_2nd.data[secname][keyname]
                        x, res = rel_diff1d(x1, y1, x2, y2, ref=1)
                        joint.data[secname][xname] = x
                    elif ndim == 2:
                        z1 = self.data[secname][keyname]
                        z2 = code_2nd.data[secname][keyname]
                        xname, yname = \
                            self._get_x_and_y_names(secname, z1.shape)
                        x1 = self.data[secname][xname]
                        y1 = self.data[secname][yname]
                        x2 = code_2nd.data[secname][xname]
                        y2 = code_2nd.data[secname][yname]
                        x, y, res = rel_diff2d(x1, y1, z1, x2, y2, z2, ref=1)
                        joint.data[secname][xname] = x
                        joint.data[secname][yname] = y
                    else:
                        raise ValueError('Array {} in {} with {} '
                                         'dimensions. Maximum 2 allowed'
                                         ''.format(keyname, secname, ndim))
                    joint.data[secname][keyname] = res
        joint.read_output = True
        return joint

    def only_here(self, code_2nd):
        """
        Return a Code class that contain the the sections/keys that
        are only in this Code (w.r.t. the second one).
        """
        # Preliminary checks
        if not self.name:
            self.name = 'unnamed_code'
        for code in [self, code_2nd]:
            if not code.read_output:
                raise IOError('For Code {} you have to load the data first! '
                              '(with load_data)'.format(code.name))

        # Initialize new code
        only = Code()
        name = 'only_in_{}'.format(self.name)
        only.name = name
        if self.path_plots:
            only.path_plots = self.path_plots

        # Copy reference keys to only ones and remove common ones
        only.keys = copy.deepcopy(self.keys)
        for sec in self.sections:
            if sec in code_2nd.sections:
                for key in self.keys[sec]:
                    if not self._is_indep_var(key):
                        if key in code_2nd.keys[sec]:
                            only.keys[sec].remove(key)
        # Remove items if only _indep_vars
        for sec in list(only.keys.keys()):
            only_indep = list(set(only.keys[sec]) - set(only._indep_vars))
            if not only_indep:
                only.keys.pop(sec)

        # Get sections
        only.sections = list(only.keys.keys())

        # Copy data
        for sec in only.sections:
            only.data[sec] = {}
            for key in only.keys[sec]:
                only.data[sec][key] = copy.deepcopy(self.data[sec][key])

        only.read_output = True
        return only

    def plot(self, other, diff=None, save=False, default_scales=None):

        # Which plots
        if diff:
            keys_to_plot = copy.deepcopy(diff.keys)
        else:
            keys_to_plot = copy.deepcopy(self.diff_codes(other).keys)

        # Create folder
        if save:
            try:
                os.mkdir(self.path_plots)
            except FileExistsError:
                pass

        # Main loop
        for sec in keys_to_plot:
            for key in keys_to_plot[sec]:
                if not self._is_indep_var(key):
                    ndim = self.data[sec][key].ndim
                    if ndim == 1:
                        # Values
                        xname = self._get_x_name(sec)
                        v1 = self.data[sec][xname], self.data[sec][key]
                        v2 = other.data[sec][xname], other.data[sec][key]
                        if diff:
                            vd = diff.data[sec][xname], diff.data[sec][key]
                        else:
                            vd = None
                        # Scales
                        try:
                            xscale = default_scales[sec][xname]
                        except KeyError:
                            xscale = 'linear'
                        try:
                            yscale = default_scales[sec][key]
                        except KeyError:
                            yscale = 'linear'
                        # Save
                        if save:
                            name = '{}_{}_of_{}.pdf'.format(sec, key, xname)
                            save_loc = os.path.join(self.path_plots, name)
                        else:
                            save_loc = None
                        # Labels
                        xlabel = xname
                        ylabel = key
                        title = sec
                        v1label = self.name
                        v2label = other.name
                        # Plot
                        plot1D(v1, v2, vd, v1label, v2label, xscale, yscale,
                               xlabel, ylabel, title, save_loc)
                        plt.close()

                    elif ndim == 2:
                        # Values
                        z1 = self.data[sec][key]
                        z2 = other.data[sec][key]
                        xname, yname = \
                            self._get_x_and_y_names(sec, z1.shape)
                        x1 = self.data[sec][xname]
                        y1 = self.data[sec][yname]
                        x2 = other.data[sec][xname]
                        y2 = other.data[sec][yname]
                        v1 = x1, y1, z1
                        v2 = x2, y2, z2
                        if diff:
                            xd = diff.data[sec][xname]
                            yd = diff.data[sec][yname]
                            zd = diff.data[sec][key]
                            vd = xd, yd, zd
                        else:
                            vd = None
                        # Scales
                        try:
                            xscale = default_scales[sec][xname]
                        except KeyError:
                            xscale = 'linear'
                        try:
                            yscale = default_scales[sec][yname]
                        except KeyError:
                            yscale = 'linear'
                        try:
                            zscale = default_scales[sec][key]
                        except KeyError:
                            zscale = 'linear'
                        # Save
                        if save:
                            name = '{}_{}_of_{}_and_{}.pdf' \
                                ''.format(sec, key, xname, yname)
                            save_loc = os.path.join(self.path_plots, name)
                        else:
                            save_loc = None
                        # Labels
                        xlabel = xname
                        ylabel = yname
                        zlabel = key
                        title = sec
                        v1label = self.name
                        v2label = other.name
                        # Plot
                        plot2D(v1, v2, vd, v1label, v2label, xscale, yscale,
                               zscale, xlabel, ylabel, zlabel, title, save_loc)
                        plt.close()

                    else:
                        raise ValueError('Array {} in {} with {} '
                                         'dimensions. Maximum 2 allowed'
                                         ''.format(key, sec, ndim))

        return


# ------------------- Plots --------------------------------------------------#


def plot1D(v1, v2, vd, v1label, v2label, xscale, yscale, xlabel, ylabel, title,
           save_loc):

    # Create figure
    plt.figure(1, figsize=(6, 6))
    plt.tight_layout()

    # Expand variables
    x1, y1 = v1
    x2, y2 = v2
    if vd:
        xd, yd = vd

    if vd:
        # First subplot
        plt.subplot(211)

    plt.plot(x1, y1, '-', label=v1label)
    plt.plot(x2, y2, '--', label=v2label)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')

    if vd:
        # Second subplot
        plt.subplot(212)

        plt.plot(xd, 100*np.abs(yd), '-')
        plt.xscale(xscale)
        plt.xlabel(xlabel)
        plt.ylabel('|diff| [%]')

        # Plots adjustements

        # Adjust separations between plots
        plt.subplots_adjust(hspace=.0)

        plt.figure(1)
        # First sublopt
        plt.subplot(211)
        ax = plt.gca()
        ax.set_xticklabels([])

    if save_loc:
        plt.savefig(save_loc, bbox_inches='tight')
    else:
        plt.show()

    return


def plot2D(v1, v2, vd, v1label, v2label, xscale, yscale, zscale, xlabel,
           ylabel, zlabel, title, save_loc):

    # Expand variables
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    if vd:
        xd, yd, zd = vd

    # TODO: For now I am just printing at z=0
    ny = 0
    new_title = '{}_at_{}_{}'.format(title, ylabel, int(y1[ny]))

    vv1 = x1, z1[:, ny]
    vv2 = x2, z2[:, ny]
    if vd:
        vvd = x2, z2[:, ny]
    plot1D(vv1, vv2, vvd, v1label, v2label, xscale, zscale, xlabel, zlabel,
           new_title, save_loc)

    return
