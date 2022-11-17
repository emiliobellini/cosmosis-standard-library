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
    parser.add_argument('--path_plots', '-p', type=str, help='Plots folder.')
    parser.add_argument('--save_plots', '-s', action='store_true', help='Save '
                        'plots without showing them (default: False).')
    parser.add_argument('--type', '-t', type=str, help='Type of output to be '
                        'plotted (e.g. matter_power_lin. distances). If left '
                        'blank all the plots will be generated.')
    parser.add_argument('--key', '-k', type=str, help='key to plot. '
                        'If left blank all the plots will be generated.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        default=False, help='Regulate verbosity.')

    return parser.parse_args()


# ------------------- Relative differences -----------------------------------#

def rel_diff1d(x1, y1, x2, y2, N=3e3, scale='linear', eps=1e-5, ref=1):
    """ Relative difference between data vectors

    Args:
        x1,y1,x2,y2 (np array): data to interpolate.
        N (int): number of points on which to interpolate.
        scale (str): how to populate the interpolation (log or linear).
        eps (float): cut the interval to avoid interpolation problems.
        ref (int): if 0 the rel diff is calculated using N points with linear
            or log spacing, if 1 (2) the rel diff is calculated only at the
            points of model 1 (2).

    Returns:
        (x,y): arrays with the relative difference.

    """

    # don't compute it many times, slow for long arrays
    xmin = (1.+0*eps)*max(min(x1), min(x2))
    xmax = (1.-0*eps)*min(max(x1), max(x2))

    data1 = interpolate.interp1d(x1, y1)
    data2 = interpolate.interp1d(x2, y2)

    if ref == 1:
        range = np.array([x for x in x1 if xmin <= x <= xmax])
    elif ref == 2:
        range = np.array([x for x in x2 if xmin <= x <= xmax])
    else:
        if scale == 'linear':
            range = np.linspace(xmin, xmax, int(N))
        else:
            range = np.exp(np.linspace(np.log(xmin), np.log(xmax), int(N)))

    d1 = np.array([data1(x) for x in range])
    d2 = np.array([data2(x) for x in range])
    diff = np.zeros_like(d1)
    for nx, x in enumerate(range):
        if d1[nx] != d2[nx]:
            diff[nx] = d2[nx]/d1[nx]-1.
    return range, diff


def rel_diff2d(x1, y1, z1, x2, y2, z2, N=3e2, scale='linear', eps=1e-5, ref=1):
    """ Relative difference between data vectors

    Args:
        x1,y1,z1,x2,y2,z2 (np array): data to interpolate.
        N (int): number of points on which to interpolate.
        spacing (str): how to populate the interpolation (log or linear).
        eps (float): cut the interval to avoid interpolation problems.
        ref (int): if 0 the rel diff is calculated using N points with linear
            or log spacing, if 1 (2) the rel diff is calculated only at the
            points of model 1 (2).

    Returns:
        (x,y,z): arrays with the relative difference.

    """

    # don't compute it many times, slow for long arrays
    xmin = (1.+0*eps)*max(min(x1), min(x2))
    xmax = (1.-0*eps)*min(max(x1), max(x2))
    ymin = (1.+0*eps)*max(min(y1), min(y2))
    ymax = (1.-0*eps)*min(max(y1), max(y2))

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
        if scale == 'linear':
            range_x = np.linspace(xmin, xmax, int(N))
            range_y = np.linspace(ymin, ymax, int(N))
        else:
            range_x = np.exp(np.linspace(np.log(xmin), np.log(xmax), int(N)))
            range_y = np.exp(np.linspace(np.log(ymin), np.log(ymax), int(N)))

    diff = np.zeros((range_x.shape[0], range_y.shape[0]))
    for nx, x in enumerate(range_x):
        d1 = np.array([data1(x, y) for y in range_y])
        d2 = np.array([data2(x, y) for y in range_y])
        for ny, y in enumerate(range_y):
            if d1[ny] != d2[ny]:
                diff[nx, ny] = d2[ny]/d1[ny]-1.
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
        # Plots
        self.vars_plots = {}
        self.scales_plots = {}
        return None

    def _initialize_with_ini(self, path_ini):
        """
        Initialize Class attributes with ini_file
        """
        if not os.path.isfile(path_ini):
            raise IOError('Ini file not found at {}'.format(path_ini))
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

    def _get_dict_plot(self, sec, key, ndim, code=None):
        v = {}
        if code:
            tp = code
        else:
            tp = self
        try:
            v['key'] = tp.data[sec][key]
        except KeyError:
            return
        if ndim == 1:
            xname = self._get_x_name(sec)
        elif ndim == 2:
            xname, yname = \
                self._get_x_and_y_names(sec, v['key'].shape, code=code)
            v['y'] = tp.data[sec][yname]
        v['x'] = tp.data[sec][xname]
        # Scales
        try:
            v['xscale'] = tp.scales_plots[sec][xname]
        except KeyError:
            v['xscale'] = 'linear'
        try:
            v['keyscale'] = tp.scales_plots[sec][key]
        except KeyError:
            v['keyscale'] = 'linear'
        # Labels
        v['xlabel'] = xname
        v['keylabel'] = key
        v['title'] = sec
        v['name'] = tp.name

        if ndim == 2:
            try:
                v['yscale'] = tp.scales_plots[sec][yname]
            except KeyError:
                v['yscale'] = 'linear'
            v['ylabel'] = yname

        return v

    def _get_x_name(self, secname):
        set1 = set(self._indep_vars)
        set2 = set(self.data[secname].keys())
        inter = list(set1 & set2)
        if len(inter) == 1:
            return inter[0]
        else:
            raise IOError('Incorrect number of dimensions. Wanted 1, '
                          'got {}'.format(len(inter)))

    def _get_x_and_y_names(self, secname, zshape, code=None):
        if code:
            tp = code
        else:
            tp = self
        set1 = set(tp._indep_vars)
        set2 = set(tp.data[secname].keys())
        inter = list(set1 & set2)
        if len(inter) == 2:
            if len(tp.data[secname][inter[0]]) == zshape[0]:
                return inter[0], inter[1]
            else:
                return inter[1], inter[0]
        else:
            raise IOError('Incorrect number of dimensions. Wanted 2, '
                          'got {}'.format(len(inter)))

    def _warn(self, msg):
        w = self._print_bf('----> WARNING:')
        print(w + ' ' + msg)
        return

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
            self._warn('output path {} already existent. '
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
                        xname_ref, yname_ref = \
                            self._get_x_and_y_names(secname, z1.shape,
                                                    code=None)
                        xname_2nd, yname_2nd = \
                            self._get_x_and_y_names(secname, z2.shape,
                                                    code=code_2nd)
                        x1 = self.data[secname][xname_ref]
                        y1 = self.data[secname][yname_ref]
                        x2 = code_2nd.data[secname][xname_2nd]
                        y2 = code_2nd.data[secname][yname_2nd]
                        x, y, res = rel_diff2d(x1, y1, z1, x2, y2, z2, ref=1)
                        joint.data[secname][xname_ref] = x
                        joint.data[secname][yname_ref] = y
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

    def setup_plots(self, args, scales):
        # Save
        if args.path_plots:
            self.path_plots = args.path_plots
        # Variables and checks
        if args.type and not args.key:
            self._warn('missing key but type specified. '
                       'Plotting all the variables!')
        elif args.key and not args.type:
            self._warn('missing type but key specified. '
                       'Plotting all the variables!')
        elif args.type and args.key:
            self.vars_plots[args.type] = [args.key]
        self.scales_plots = scales
        return

    def plot(self, args, other=None, diff=None, scales=None):

        # Setup (which plots)
        self.setup_plots(args, scales)
        if not self.vars_plots:
            if diff:
                keys = diff.keys
            else:
                keys = self.keys
            for sec in keys.keys():
                self.vars_plots[sec] = []
                for key in keys[sec]:
                    if not self._is_indep_var(key):
                        self.vars_plots[sec].append(key)

        # Create folder
        if args.save_plots:
            try:
                os.mkdir(self.path_plots)
            except FileExistsError:
                pass

        # Main loop
        for sec in self.vars_plots.keys():
            for key in self.vars_plots[sec]:
                ndim = self.data[sec][key].ndim
                v1 = self._get_dict_plot(sec, key, ndim, code=None)
                if other:
                    v2 = self._get_dict_plot(sec, key, ndim, code=other)
                else:
                    v2 = None
                if diff:
                    vd = self._get_dict_plot(sec, key, ndim, code=diff)
                else:
                    vd = None
                if ndim == 1:
                    plot1D(v1, v2, vd)
                elif ndim == 2:
                    plot2D(v1, v2, vd)
                else:
                    raise ValueError('Array {} in {} with {} '
                                     'dimensions. Maximum 2 allowed'
                                     ''.format(key, sec, ndim))
                if args.save_plots:
                    fname = '{}_{}.pdf'.format(sec, key)
                    save_loc = os.path.join(self.path_plots, fname)
                    plt.savefig(save_loc, bbox_inches='tight')
                else:
                    plt.show()
                plt.close()

        return


# ------------------- Plots --------------------------------------------------#

def plot1D(v1, v2, vd):

    # Plot limits
    if v2:
        xlimmin = min(v1['x'].min(), v2['x'].min())
        xlimmax = max(v1['x'].max(), v2['x'].max())
    else:
        xlimmin, xlimmax = v1['x'].min(), v1['x'].max()

    # Create figure
    if vd:
        sizey = 4
    else:
        sizey = 3
    plt.figure(1, figsize=(6, sizey))
    plt.tight_layout()

    if vd:
        # First subplot
        plt.subplot(211)

    plt.plot(v1['x'], v1['key'], '-', label=v1['name'])
    if v2:
        try:
            plt.plot(v2['x'], v2['key'], '--', label=v2['name'])
        except KeyError:
            pass

    plt.xlim(xlimmin, xlimmax)
    plt.xscale(v1['xscale'])
    plt.yscale(v1['keyscale'])
    plt.ylabel(v1['keylabel'])
    plt.title(v1['title'])
    plt.legend(loc='best')

    if vd:
        # Second subplot
        plt.subplot(212)

        plt.plot(vd['x'], 100*np.abs(vd['key']), 'k-')
        plt.xlim(xlimmin, xlimmax)
        plt.xscale(v1['xscale'])
        plt.xlabel(v1['xlabel'])
        plt.ylabel('|diff| [%]')

        # Adjust separations between plots
        plt.subplots_adjust(hspace=.0)

        plt.figure(1)
        # First sublopt
        plt.subplot(211)
        ax = plt.gca()
        ax.set_xticklabels([])

    return


def plot2D(v1, v2, vd):

    # Preliminary checks and adjustments
    if v2:
        if v2['xlabel'] == v1['ylabel']:
            v2['xlabel_tmp'] = v2['xlabel']
            v2['xlabel'] = v2['ylabel']
            v2['ylabel'] = v2['xlabel_tmp']
            v2.pop('xlabel_tmp')
            v2['xscale_tmp'] = v2['xscale']
            v2['xscale'] = v2['yscale']
            v2['yscale'] = v2['xscale_tmp']
            v2.pop('xscale_tmp')
            v2['x_tmp'] = copy.deepcopy(v2['x'])
            v2['x'] = copy.deepcopy(v2['y'])
            v2['y'] = copy.deepcopy(v2['x_tmp'])
            v2.pop('x_tmp')
    if vd:
        if vd['xlabel'] == v1['ylabel']:
            vd['xlabel_tmp'] = vd['xlabel']
            vd['xlabel'] = vd['ylabel']
            vd['ylabel'] = vd['xlabel_tmp']
            vd.pop('xlabel_tmp')
            vd['xscale_tmp'] = vd['xscale']
            vd['xscale'] = vd['yscale']
            vd['yscale'] = vd['xscale_tmp']
            vd.pop('xscale_tmp')
            vd['x_tmp'] = copy.deepcopy(vd['x'])
            vd['x'] = copy.deepcopy(vd['y'])
            vd['y'] = copy.deepcopy(vd['x_tmp'])
            vd.pop('x_tmp')

    # Create figure
    if vd:
        sizey = 8
    else:
        sizey = 6
    plt.figure(1, figsize=(12, sizey))
    plt.tight_layout()

    # Plot limits
    if vd:
        xmin, xmax = vd['x'].min(), vd['x'].max()
        ymin, ymax = vd['y'].min(), vd['y'].max()
    elif v2:
        xmin = max(v1['x'].min(), v2['x'].min())
        xmax = min(v1['x'].max(), v2['x'].max())
        ymin = max(v1['y'].min(), v2['y'].min())
        ymax = min(v1['y'].max(), v2['y'].max())
    else:
        xmin, xmax = v1['x'].min(), v1['x'].max()
        ymin, ymax = v1['y'].min(), v1['y'].max()
    if v2:
        xlimmin = min(v1['x'].min(), v2['x'].min())
        xlimmax = max(v1['x'].max(), v2['x'].max())
        ylimmin = min(v1['y'].min(), v2['y'].min())
        ylimmax = max(v1['y'].max(), v2['y'].max())
    else:
        xlimmin, xlimmax = xmin, xmax
        ylimmin, ylimmax = ymin, ymax

    # First subplot
    plt.subplot(221)

    d1 = np.array([interpolate.interp1d(v1['y'], v1['key'][nx])
                  for nx, x in enumerate(v1['x'])])
    d1_max = np.array([d(ymax) for d in d1])
    d1_min = np.array([d(ymin) for d in d1])
    d1_label_min = '{}_{}_{:.2e}'.format(v1['name'], v1['ylabel'], ymin)
    d1_label_max = '{}_{}_{:.2e}'.format(v1['name'], v1['ylabel'], ymax)

    plt.plot(v1['x'], d1_min, '-', label=d1_label_min)
    plt.plot(v1['x'], d1_max, '-', label=d1_label_max)

    if v2:
        d2 = np.array([interpolate.interp1d(v2['y'], v2['key'][nx])
                      for nx, x in enumerate(v2['x'])])
        d2_max = np.array([d(ymax) for d in d2])
        d2_min = np.array([d(ymin) for d in d2])
        d2_label_min = '{}_{}_{:.2e}'.format(v2['name'], v2['ylabel'], ymin)
        d2_label_max = '{}_{}_{:.2e}'.format(v2['name'], v2['ylabel'], ymax)

        plt.plot(v2['x'], d2_min, ':', label=d2_label_min)
        plt.plot(v2['x'], d2_max, ':', label=d2_label_max)

    plt.xlim(xlimmin, xlimmax)
    plt.xscale(v1['xscale'])
    plt.yscale(v1['keyscale'])
    plt.ylabel(v1['keylabel'])
    plt.title(v1['title'])
    plt.legend(loc='best')

    # Second subplot
    plt.subplot(222)

    d1 = np.array([interpolate.interp1d(v1['x'], v1['key'][:, ny])
                  for ny, y in enumerate(v1['y'])])
    d1_max = np.array([d(xmax) for d in d1])
    d1_min = np.array([d(xmin) for d in d1])
    d1_label_min = '{}_{}_{:.2e}'.format(v1['name'], v1['xlabel'], xmin)
    d1_label_max = '{}_{}_{:.2e}'.format(v1['name'], v1['xlabel'], xmax)

    plt.plot(v1['y'], d1_min, '-', label=d1_label_min)
    plt.plot(v1['y'], d1_max, '-', label=d1_label_max)

    if v2:
        d2 = np.array([interpolate.interp1d(v2['x'], v2['key'][:, ny])
                      for ny, y in enumerate(v2['y'])])
        d2_max = np.array([d(xmax) for d in d2])
        d2_min = np.array([d(xmin) for d in d2])
        d2_label_min = '{}_{}_{:.2e}'.format(v2['name'], v2['xlabel'], xmin)
        d2_label_max = '{}_{}_{:.2e}'.format(v2['name'], v2['xlabel'], xmax)

        plt.plot(v2['y'], d2_min, ':', label=d2_label_min)
        plt.plot(v2['y'], d2_max, ':', label=d2_label_max)

    plt.xlim(ylimmin, ylimmax)
    plt.xscale(v1['yscale'])
    plt.yscale(v1['keyscale'])
    plt.ylabel(v1['keylabel'])
    plt.title(v1['title'])
    plt.legend(loc='best')

    if vd:
        # Third subplot
        plt.subplot(223)

        dd = np.array([interpolate.interp1d(vd['y'], vd['key'][nx])
                      for nx, x in enumerate(vd['x'])])
        dd_max = np.array([d(ymax) for d in dd])
        dd_min = np.array([d(ymin) for d in dd])
        dd_label_min = '{}_{}_{:.2e}'.format(vd['name'], vd['ylabel'], ymin)
        dd_label_max = '{}_{}_{:.2e}'.format(vd['name'], vd['ylabel'], ymax)

        plt.plot(vd['x'], 100*np.abs(dd_min), '-', label=dd_label_min)
        plt.plot(vd['x'], 100*np.abs(dd_max), '--', label=dd_label_max)

        plt.xlim(xlimmin, xlimmax)
        plt.xscale(v1['xscale'])
        plt.xlabel(v1['xlabel'])
        plt.ylabel('|diff| [%]')
        plt.legend(loc='best')

        # Fourth subplot
        plt.subplot(224)

        dd = np.array([interpolate.interp1d(vd['x'], vd['key'][:, ny])
                      for ny, y in enumerate(vd['y'])])
        dd_max = np.array([d(xmax) for d in dd])
        dd_min = np.array([d(xmin) for d in dd])
        dd_label_min = '{}_{}_{:.2e}'.format(vd['name'], vd['xlabel'], xmin)
        dd_label_max = '{}_{}_{:.2e}'.format(vd['name'], vd['xlabel'], xmax)

        plt.plot(vd['y'], 100*np.abs(dd_min), '-', label=dd_label_min)
        plt.plot(vd['y'], 100*np.abs(dd_max), '--', label=dd_label_max)

        plt.xlim(ylimmin, ylimmax)
        plt.xscale(v1['yscale'])
        plt.xlabel(vd['ylabel'])
        plt.ylabel('|diff| [%]')
        plt.legend(loc='best')

        # Adjust separations between plots
        plt.subplots_adjust(hspace=.0)

        plt.figure(1)
        # First sublopt
        plt.subplot(221)
        ax = plt.gca()
        ax.set_xticklabels([])
        # Second sublopt
        plt.subplot(222)
        ax = plt.gca()
        ax.set_xticklabels([])

    return
