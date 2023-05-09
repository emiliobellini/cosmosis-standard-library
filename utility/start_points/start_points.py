from cosmosis.datablock import option_section
from cosmosis.runtime.parameter import Parameter
import numpy as np
import re


def setup(options):

    if options['runtime', 'sampler'] != 'emcee':
        raise IOError('start_points works only with the Emcee sampler!')

    config = {
        'mean': options.get_string(option_section, 'mean', default=''),
        'covmat': options.get_string(option_section, 'covmat', default=''),
        'squeeze_factor': options.get_double(
            option_section, 'squeeze_factor', default=10.0),
        'debug': options.get_bool(option_section, 'debug', default=False),
        'debug_samples': options.get_int(
            option_section, 'debug_samples', default=1000),
        'rescale': options.get_bool(option_section, 'rescale', default=True),
        'truncate': options.get_bool(option_section, 'truncate', default=True),
    }

    n_walkers = options.get_int('emcee', 'walkers')
    values_fn = options.get_string('pipeline', 'values')
    start_points_fn = options.get_string('emcee', 'start_points')

    # Initialize all parameters and get varying ones
    params = Parameter.load_parameters(values_fn)
    varied_params = [p for p in params if p.is_varied()]

    # Build means array
    mean_dict = read_means(config['mean'])
    mean = []
    for p in varied_params:
        try:
            mean.append(mean_dict[p.section][p.name])
        except KeyError:
            mean.append(p.start)
    mean = np.array(mean)

    # Build covmat
    covmat = np.zeros((len(varied_params), len(varied_params)))
    covmat_keys, covmat_values = read_covmat(config['covmat'])
    for n1, p1 in enumerate(varied_params):
        for n2, p2 in enumerate(varied_params):
            try:
                idx1 = covmat_keys.index((p1.section, p1.name))
                idx2 = covmat_keys.index((p2.section, p2.name))
                covmat[n1, n2] = covmat_values[idx1, idx2]
            except ValueError:
                if n1 == n2:
                    covmat[n1, n1] = (p1.width()/config['squeeze_factor'])**2.

    # Get random points
    start_points = get_start_points(
        varied_params, mean, covmat, n_walkers,
        rescale=config['rescale'], truncate=config['truncate'])

    # Save to file
    header = ['{}--{}'.format(x.section, x.name) for x in varied_params]
    header = '\t'.join(header)
    np.savetxt(start_points_fn, start_points, header=header)

    if config['debug']:
        def msg(data, head):
            print('------> {}. Max: {:.2e},  Average: {:.2e}, # Above '
                  'threshold ({:.2e}): {}'
                  ''.format(head, np.max(data), np.average(data), threshold,
                            len(data[data > threshold])))

        threshold = 1.e-2
        n_samples = config['debug_samples']
        print('Module start_points. Debug info for {} samples:'
              ''.format(n_samples))
        for res, tru in [(False, False), (True, False), (True, True)]:
            if res:
                msg_r = 'rescale'
            else:
                msg_r = 'no rescale'
            if tru:
                msg_t = 'truncate'
            else:
                msg_t = 'no truncate'
            print('----> {}, {}. Relative (abs) deviations:'
                  ''.format(msg_r, msg_t))
            points = get_start_points(
                varied_params, mean, covmat, n_samples,
                rescale=res, truncate=tru)
            mean_rt = np.abs(np.mean(points, axis=0)/mean-1.)
            msg(mean_rt, 'Mean')
            cov_rt = np.abs(np.cov(points.T)/covmat-1.)
            msg(np.diag(cov_rt), 'Cov (diag)')
            idx1, idx2 = np.triu_indices(3, k=1)
            msg(cov_rt[idx1, idx2], 'Cov (off diag)')

    return config


def execute(block, config):
    return 0


def cleanup(config):
    pass


def read_means(path):
    if path == '':
        return {}
    with open(path) as fn:
        lines = fn.readlines()
    means = {}
    for line in lines:
        line = re.sub('#.*', '', line)
        line = line.split(' ')
        line = list(filter(None, line))
        line = [x.strip() for x in line]
        if len(line) == 2:
            name = line[0].split('--')
            if len(name) == 2:
                sec, param = line[0].split('--')
                value = float(line[1])
                try:
                    means[sec][param] = value
                except KeyError:
                    means[sec] = {}
                    means[sec][param] = value
    return means


def read_covmat(path):
    if path == '':
        return [], np.array([])
    with open(path) as fn:
        line = fn.readline()
    keys = line.strip().strip('#')
    keys = keys.split(' ')
    keys = list(filter(None, keys))
    keys = [tuple(x.split('--')) for x in keys]
    values = np.genfromtxt(path)
    if values.shape != (len(keys), len(keys)):
        raise ValueError('Inconsistent dimensions!')
    return keys, values


def get_start_points(params, mean, covmat, n_samples,
                     rescale=True, truncate=True):
    start_points = np.zeros((n_samples, len(mean)))
    if rescale:
        factor = np.diag(1./np.sqrt(np.diag(covmat)))
        inv_factor = np.diag(np.sqrt(np.diag(covmat)))
        rescaled_mean = factor.dot(mean)
        rescaled_covmat = factor.T.dot(covmat).dot(factor)
    else:
        rescaled_mean = mean
        rescaled_covmat = covmat
    count = 0
    while count < n_samples:
        point = np.random.multivariate_normal(rescaled_mean, rescaled_covmat)
        if rescale:
            point = point.dot(inv_factor)
        if truncate:
            in_range = [p.in_range(point[n]) for n, p in enumerate(params)]
            if all(in_range):
                start_points[count, :] = point
                count += 1
        else:
            start_points[count, :] = point
            count += 1
    return start_points
