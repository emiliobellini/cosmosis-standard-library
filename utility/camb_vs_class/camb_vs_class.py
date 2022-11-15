import camb_vs_class_tools as tools

default_scales = {
    'cmb_cl': {
        'ell': 'log',
        'ee': 'linear',
        'bb': 'linear',
        'pe': 'linear',
        'pp': 'linear',
        'pt': 'linear',
        'te': 'linear',
        'tt': 'linear',
    },
    'distances': {
        'a': 'log',
        'd_a': 'linear',
        'd_l': 'linear',
        'd_m': 'linear',
        'f_ap': 'linear',
        'h': 'linear',
        'mu': 'linear',
        'rs_dv': 'linear',
        'z': 'log',
    },
    'growth_parameters': {
        'd_z': 'linear',
        'da': 'linear',
        'f_ap': 'linear',
        'f_z': 'linear',
        'fsigma_8': 'linear',
        'h': 'linear',
        'rs_dv': 'linear',
        'sigma_8': 'linear',
        'z': 'log',
    },
    'matter_power_lin': {
        'k_h': 'log',
        'p_k': 'log',
        'z': 'linear',
    },
    'matter_power_nl': {
        'k_h': 'log',
        'p_k': 'log',
        'z': 'linear',
    },
}


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Call the parser
    args = tools.argument_parser()

    # Initialize reference code
    code_ref = tools.Code(args.params_ref)
    code_ref.run()
    code_ref.load_data()
    if args.verbose:
        code_ref.print_keys()

    # Initialize second code
    code_2nd = tools.Code(args.params_2nd)
    code_2nd.run()
    code_2nd.load_data()
    if args.verbose:
        code_2nd.print_keys()

    # Get difference between two codes
    diff = code_ref.diff_codes(code_2nd)
    if args.verbose:
        diff.print_keys()

    # Isolate data that are only in the reference code
    only_ref = code_ref.only_here(code_2nd)
    if args.verbose:
        only_ref.print_keys()

    # Plots
    code_ref.plot(args, other=code_2nd, diff=diff, scales=default_scales)
    # code_ref.plot(args, other=None, diff=None, scales=default_scales)
