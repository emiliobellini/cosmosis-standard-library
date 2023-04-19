import check_convergence_tools as tools


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Call the parser
    args = tools.argument_parser()

    # Ini file
    ini = tools.IniFile(args.params_ini).read()

    if ini['runtime']['sampler'] != 'emcee':
        raise ValueError('Currently implemented only for emcee')

    n_walkers = int(ini['emcee']['walkers'])

    # Import chains
    chain = tools.ChainFile(args.chain_file).read().reshape(n_walkers)

    # Get autocorrelations
    chain.estimate_autocorrelations(num=args.num_auto_points, plots=args.plots)
