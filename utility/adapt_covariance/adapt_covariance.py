from cosmosis.datablock import option_section
from cosmosis.runtime.parameter import Parameter
import numpy as np


def setup(options):

    sampler = options.get_string('runtime', 'sampler')
    config = {
        'input_covmat': options.get_string(option_section, 'input_covmat'),
        'output_covmat': options.get_string(sampler, 'covmat'),
        'squeeze_factor': options.get_double(
            option_section, 'squeeze_factor', default=10.0),
    }

    values_fn = options.get_string('pipeline', 'values')

    # Initialize all parameters and get varying ones
    params = Parameter.load_parameters(values_fn)
    varied_params = [p for p in params if p.is_varied()]

    # Build covmat
    output_covmat = np.zeros((len(varied_params), len(varied_params)))
    input_covmat_keys, input_covmat_values = read_covmat(
        config['input_covmat'])
    for n1, p1 in enumerate(varied_params):
        for n2, p2 in enumerate(varied_params):
            try:
                idx1 = input_covmat_keys.index((p1.section, p1.name))
                idx2 = input_covmat_keys.index((p2.section, p2.name))
                output_covmat[n1, n2] = input_covmat_values[idx1, idx2]
            except ValueError:
                if n1 == n2:
                    output_covmat[n1, n1] = \
                        (p1.width()/config['squeeze_factor'])**2.

    # Save to file
    header = ['{}--{}'.format(x.section, x.name) for x in varied_params]
    header = '\t'.join(header)
    np.savetxt(config['output_covmat'], output_covmat, header=header)

    return config


def execute(block, config):
    return 0


def cleanup(config):
    pass


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
