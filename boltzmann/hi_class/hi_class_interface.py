import os
import sys
import traceback
import warnings
import numpy as np
from builtins import str
from cosmosis.datablock import names, option_section
from cosmosis.runtime.config import Inifile

# add class directory to the path
dirname = os.path.split(__file__)[0]
# enable debugging from the same directory
if not dirname.strip():
    dirname = '.'

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters
distances = names.distances
cmb_cl = names.cmb_cl


def setup(options):
    pyversion = f"{sys.version_info.major}.{sys.version_info.minor}"
    class_dir = "hi_class_pub_devel"
    lib_dir = f"classy_install/lib/python{pyversion}/site-packages"
    install_dir = os.path.join(dirname, class_dir, lib_dir)
    with open(f"{install_dir}/easy-install.pth") as f:
        pth = f.read().strip()
        install_dir = os.path.join(install_dir, pth)

    sys.path.insert(0, install_dir)

    import classy
    print(f"Loaded classy from {classy.__file__}")

    # Read options from the ini file which are fixed across
    # the length of the chain
    config = {
        'lmax': options.get_int(option_section, 'lmax', default=2500),
        'zmax': options.get_double(option_section, 'zmax', default=4.0),
        'kmax': options.get_double(option_section, 'kmax', default=50.),
        'nk': options.get_int(option_section, 'nk', default=100),
        'debug': options.get_bool(option_section, 'debug', default=False),
        'lensing': options.get_bool(option_section, 'lensing', default=True),
        'cmb': options.get_bool(option_section, 'cmb', default=True),
        'mpk': options.get_bool(option_section, 'mpk', default=True),
        'save_matter_power_lin': options.get_bool(
            option_section, 'save_matter_power_lin', default=True),
        'save_cdm_baryon_power_lin': options.get_bool(
            option_section, 'save_cdm_baryon_power_lin', default=False),
        'save_errors': options.get_bool(
            option_section, 'save_errors', default=False),
    }

    # HI_CLASS_NEW: if save_errors create a file with the errors
    if config['save_errors']:
        ini_values = Inifile(options['pipeline', 'values'])
        config['varying_params'] = get_varying_params(ini_values)
        name, ext = os.path.splitext(options['output', 'filename'])
        name += '_errors'
        config['save_errors_file'] = name + ext
        parent_folder, _ = os.path.split(config['save_errors_file'])
        if not os.path.isdir(parent_folder):
            os.makedirs(parent_folder)
        if not os.path.isfile(config['save_errors_file']):
            fe = open(config['save_errors_file'], 'w')
            fe.write('#{}\n'.format(
                '\t'.join(['--'.join(x) for x in config['varying_params']])))
            fe.close()

    # HI_CLASS_NEW: all the configuration parameters typical of (hi_)class
    # can be specified prepending "class_" or "hi_class_" to their (hi_)class
    # names. We provide the two options for completeness, even if they are
    # equivalent. On the other hand, it is possible to add them to the config
    # dictionary. But with this approach we do not need to implement in this
    # interface all the possible parameters. Configuration parameters are those
    # that do not vary during MCMC and/or they are not floats (e.g. strings).
    # All these parameters should be written in the [hi_class] section of
    # the ini file.
    for _, key in options.keys(option_section):
        if key.startswith('class_') or key.startswith('hi_class_'):
            config[key] = options[option_section, key]
            # HI_CLASS_NEW: classy does not like True/False as inputs, we
            # need to reconvert them back to yes/no
            if isinstance(config[key], bool):
                config[key] = 'yes' if config[key] else 'no'

    # Create the object that connects to Class
    config['cosmo'] = classy.Class()

    # Return all this config information
    return config


def choose_outputs(config):
    outputs = []
    if config['cmb']:
        outputs.append("tCl pCl")
    if config['lensing']:
        outputs.append("lCl")
    if config["mpk"]:
        outputs.append("mPk")
    return " ".join(outputs)


def get_class_inputs(block, config):

    # Get parameters from block and give them the
    # names and form that class expects
    nnu = block.get_double(cosmo, 'nnu', 3.046)
    nmassive = block.get_int(cosmo, 'num_massive_neutrinos', default=0)
    params = {
        'output': choose_outputs(config),
        'lensing':   'yes' if config['lensing'] else 'no',
        'A_s':       block[cosmo, 'A_s'],
        'n_s':       block[cosmo, 'n_s'],
        'H0':        100 * block[cosmo, 'h0'],
        'omega_b':   block[cosmo, 'ombh2'],
        'omega_cdm': block[cosmo, 'omch2'],
        'tau_reio':  block[cosmo, 'tau'],
        'T_cmb':     block.get_double(cosmo, 'TCMB', default=2.726),
        'N_ur':      nnu - nmassive,
        'N_ncdm':    nmassive,
    }

    # HI_CLASS_NEW: only read m_ncdm if there are massive neutrinos
    # (bug in class interface)
    if params["N_ncdm"] > 0.:
        params.update({
          'm_ncdm': block.get_double(cosmo, 'mnu', default=0.06),
        })

    if config["cmb"] or config["lensing"]:
        params.update({
          'l_max_scalars': config["lmax"],
        })

    if config["mpk"]:
        params.update({
            'P_k_max_h/Mpc':  config["kmax"],
            'z_pk': ', '.join(str(z)
                              for z in np.arange(0.0, config['zmax'], 0.01)),
            'z_max_pk': config['zmax'],
        })

    if block.has_value(cosmo, "massless_nu"):
        warnings.warn("Parameter massless_nu is being ignored. Set nnu, the "
                      "effective number of relativistic species in the early"
                      " Universe.")

    if ((block.has_value(cosmo, "omega_nu")
            or block.has_value(cosmo, "omnuh2"))
            and not block.has_value(cosmo, "mnu")):
        warnings.warn("Parameter omega_nu and omnuh2 are being ignored. Set "
                      "mnu and num_massive_neutrinos instead.")

    # HI_CLASS_NEW: write here all the necessary cosmological parameters
    # relevant for hi_class (put them in the [cosmological_parameters] section
    # of the ini file). The other method if we do not need to vary them
    # and we do not want to modify this interface is to prepend class_ or
    # hi_class_ (they are equivalent) to their (hi_)class name (put them in the
    # [hi_class] section of the ini file).
    # IMPORTANT: the presence of a negative omega_smg is the flag to swith on
    # hi_class, otherwise normal Class will be executed. So, if the new
    # parameter is typical of hi_class put it inside the if statement, if it
    # is a standard Class parameter write it outside.
    if block.has_value(cosmo, 'omega_smg') and block[cosmo, 'omega_smg'] < 0.:
        # These are mandatory
        params['Omega_Lambda'] = block[cosmo, 'omega_lambda']
        params['Omega_smg'] = block[cosmo, 'omega_smg']
        params['Omega_fld'] = block[cosmo, 'omega_fld']
        # These are optional
        try_to_get_arrays_class(params, block, 'parameters_smg')
        try_to_get_arrays_class(params, block, 'expansion_smg')

        try_to_get_arrays_class(params, block, 'back_spline_z_smg')
        try_to_get_arrays_class(params, block, 'back_spline_domega_smg')
        try_to_get_arrays_class(params, block, 'mgclass_spline_z_smg')
        try_to_get_arrays_class(params, block, 'mgclass_spline_dmu_smg')
        try_to_get_arrays_class(params, block, 'mgclass_spline_dgamma_smg')
        try_to_get_arrays_class(params, block, 'mgclass_spline_dsigma_smg')

        try_to_get_arrays_class(params, block, 'spline_z_smg')
        try_to_get_arrays_class(params, block, 'spline_domega_smg')
        try_to_get_arrays_class(params, block, 'spline_braiding_smg')
        try_to_get_arrays_class(params, block, 'spline_running_smg')
        try_to_get_arrays_class(params, block, 'spline_dM2_smg')
        try_to_get_arrays_class(params, block, 'spline_dcs2_smg')

        try_to_get_arrays_class(params, block, 'binning_z_smg')
        try_to_get_arrays_class(params, block, 'binning_domega_smg')
        try_to_get_arrays_class(params, block, 'binning_braiding_smg')
        try_to_get_arrays_class(params, block, 'binning_running_smg')
        try_to_get_arrays_class(params, block, 'binning_dM2_smg')
        try_to_get_arrays_class(params, block, 'binning_dcs2_smg')

        # HI_CLASS_NEW: all the configuration parameters that start with
        # class_ or hi_class_ are now written in the param dictionary without
        # the prefix, to be used during the (hi_)class run.
        for key, val in config.items():
            if key.startswith('hi_class_'):
                params[key[9:]] = val

        # HI_CLASS_NEW: Adjust here config keys
        # (e.g. the parser is case insensitive).
        try_to_change_name(params, 'use_sigma', 'use_Sigma')

    for key, val in config.items():
        if key.startswith('class_'):
            params[key[6:]] = val

    # Modify params if the input is binning derivatives
    # if params['expansion_model'] == 'binning_der_domega':
    #     params['expansion_model'] = 'binning_domega'
    #     z_vec = floatify_array(params['binning_z_smg'])
    #     domega = floatify_array(params['binning_domega_smg'])
    #     r = params['binning_rel_width_smg']
    #     domega = tanh_from_derivs(z_vec, domega[0], domega[1:])
    #     params['binning_domega_smg'] = stringify_array(domega)

    # if params['gravity_model'] == 'binning_der_alphas':
    #     params['gravity_model'] = 'binning_alphas'
    #     z_vec = floatify_array(params['binning_z_smg'])
    #     bra = floatify_array(params['binning_braiding_smg'])
    #     run = floatify_array(params['binning_running_smg'])
    #     r = params['binning_rel_width_smg']
    #     bra = tanh_from_derivs(z_vec, bra[0], bra[1:])
    #     run = tanh_from_derivs(z_vec, run[0], run[1:])
    #     params['binning_braiding_smg'] = stringify_array(bra)
    #     params['binning_running_smg'] = stringify_array(run)

    # if params['gravity_model'] == 'binning_der_M2cs2':
    #     params['gravity_model'] = 'binning_M2cs2'
    #     z_vec = floatify_array(params['binning_z_smg'])
    #     dM2 = floatify_array(params['binning_dM2_smg'])
    #     dcs2 = floatify_array(params['binning_dcs2_smg'])
    #     r = params['binning_rel_width_smg']
    #     dM2 = tanh_from_derivs(z_vec, dM2[0], dM2[1:])
    #     dcs2 = tanh_from_derivs(z_vec, dcs2[0], dcs2[1:])
    #     params['binning_dM2_smg'] = stringify_array(dM2)
    #     params['binning_dcs2_smg'] = stringify_array(dcs2)

    return params


def get_class_outputs(block, c, config, params):
    ##
    # Derived cosmological parameters
    ##

    h0 = block[cosmo, 'h0']

    # Omega_smg
    # if params['expansion_model'] == 'spline_domega':
    #     z_sample = params['spline_z_smg']
    # elif params['expansion_model'] == 'binning_domega':
    #     z_sample = params['binning_z_smg']
    # elif params['expansion_model'] == 'domega_spline':
    #     z_sample = params['back_spline_z_smg']
    # z_sample = z_sample.split(',')
    # z_sample = [float(z) for z in z_sample]
    # block[cosmo, 'omega_de'] = [c.Omega_smg(z) for z in z_sample]

    ##
    # Matter power spectrum
    ##

    # Ranges of the redshift and matter power
    dz = 0.01
    kmin = 1e-4
    kmax = config['kmax'] * h0
    # nk = 100
    nk = config['nk']

    # Define k,z we want to sample
    z = np.arange(0.0, config["zmax"] + dz, dz)
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    nz = len(z)
    # HI_CLASS_NEW: k_fiducial for scale dependent growth factor
    k_fiducial = 1.e-2

    # Extract (interpolate) P(k,z) at the requested
    # sample points.
    if 'mPk' in c.pars['output']:
        block[cosmo, 'sigma_8'] = np.nan_to_num(c.sigma8())

        # Total matter power spectrum (saved as grid)
        if config['save_matter_power_lin']:
            P = np.zeros((k.size, z.size))
            for i, ki in enumerate(k):
                for j, zi in enumerate(z):
                    P[i, j] = np.nan_to_num(c.pk_lin(ki, zi))
            # HI_CLASS_NEW: inverted z and k for compatibility with camb output
            block.put_grid("matter_power_lin", "z", z, "k_h", k / h0,
                           "p_k", P.T * h0**3)

        # CDM+baryons power spectrum
        if config['save_cdm_baryon_power_lin']:
            P = np.zeros((k.size, z.size))
            for i, ki in enumerate(k):
                for j, zi in enumerate(z):
                    P[i, j] = np.nan_to_num(c.pk_cb_lin(ki, zi))
            # HI_CLASS_NEW: inverted z and k for compatibility with camb output
            block.put_grid('cdm_baryon_power_lin', 'z', z, 'k_h', k/h0,
                           'p_k', P.T*h0**3)

        # Get growth rates and sigma_8
        # D_ref = [c.scale_independent_growth_factor(zi) for zi in z]
        # f_ref = [c.scale_independent_growth_factor_f(zi) for zi in z]
        # HI_CLASS_NEW: scale dependent growth factor
        D = [np.nan_to_num(c.scale_dependent_growth_factor_at_k_and_z(
            k_fiducial, zi)) for zi in z]
        f = [np.nan_to_num(c.scale_dependent_growth_factor_f_at_k_and_z(
            k_fiducial, zi, z_step=0.1)) for zi in z]
        # HI_CLASS_NEW: uncomment to see plots of
        # relative diff between old an new growth rates
        # import matplotlib.pyplot as plt
        # plt.figure(1, figsize=(10, 8))
        # plt.subplot(211)
        # plt.plot(z, D_ref, label='back')
        # plt.plot(z, D, '--', label='pert(k={:.2e})'.format(k_fiducial))
        # plt.ylabel('D')
        # plt.legend(loc='best')
        # plt.subplot(212)
        # plt.plot(z, np.array(D)/np.array(D_ref)-1.)
        # plt.ylabel('rel_diff')
        # plt.subplots_adjust(hspace=.0)
        # plt.show()
        # plt.figure(1, figsize=(10, 8))
        # plt.subplot(211)
        # plt.plot(z, f_ref, label='back')
        # plt.plot(z, f, '--', label='pert(k={:.2e})'.format(k_fiducial))
        # plt.ylabel('f')
        # plt.legend(loc='best')
        # plt.subplot(212)
        # plt.plot(z, np.array(f)/np.array(f_ref)-1.)
        # plt.ylabel('rel_diff')
        # plt.subplots_adjust(hspace=.0)
        # plt.show()

        # fsigma = [c.effective_f_sigma8(zi) for zi in z]
        # HI_CLASS_NEW: providing R in units of Mpc/h for backward
        # compatibility
        # sigma_8_z = [c.sigma(8.0, zi, h_units=True) for zi in z]
        sigma_8_z = [np.nan_to_num(c.sigma(8.0/h0, zi)) for zi in z]
        block[names.growth_parameters, "z"] = z
        block[names.growth_parameters, "sigma_8"] = np.array(sigma_8_z)
        block[names.growth_parameters, "fsigma_8"] = \
            np.array(sigma_8_z) * np.array(f)
        block[names.growth_parameters, "d_z"] = np.array(D)
        block[names.growth_parameters, "f_z"] = np.array(f)
        block[names.growth_parameters, "a"] = 1/(1+z)

        if c.nonlinear_method != 0:
            for i, ki in enumerate(k):
                for j, zi in enumerate(z):
                    P[i, j] = np.nan_to_num(c.pk(ki, zi))

            # HI_CLASS_NEW: inverted z and k for compatibility with camb output
            block.put_grid("matter_power_nl", "z", z, "k_h", k / h0,
                           "p_k", P.T * h0**3)

    ##
    # Distances and related quantities
    ##

    # HI_CLASS_NEW: added: a, mu, H
    # save redshifts of samples
    block[distances, 'z'] = z
    block[distances, 'a'] = 1/(z+1)
    block[distances, 'nz'] = nz

    # Save distance samples
    d_l = np.array([c.luminosity_distance(zi) for zi in z])
    block[distances, 'd_l'] = d_l
    d_a = np.array([c.angular_distance(zi) for zi in z])
    block[distances, 'd_a'] = d_a
    block[distances, 'd_m'] = d_a * (1 + z)

    # Deal with mu(0), which is -np.inf
    mu = np.zeros_like(d_l)
    pos = d_l > 0
    mu[pos] = 5*np.log10(d_l[pos])+25
    mu[~pos] = -np.inf
    block[distances, 'mu'] = mu
    H = np.array([c.Hubble(zi) for zi in z])
    block[distances, 'H'] = H

    # Save some auxiliary related parameters
    block[distances, 'age'] = c.age()
    block[distances, 'rs_zdrag'] = c.rs_drag()
    block[distances, 'zdrag'] = c.z_drag()

    ##
    # Now the CMB C_ell
    ##
    if config["cmb"]:
        c_ell_data = c.lensed_cl() if config['lensing'] else c.raw_cl()
        ell = c_ell_data['ell']
        ell = ell[2:]

        # Save the ell range
        block[cmb_cl, "ell"] = ell

        # t_cmb is in K, convert to mu_K, and add ell(ell+1) factor
        tcmb_muk = block[cosmo, 'tcmb'] * 1e6
        f = ell * (ell + 1.0) / 2 / np.pi * tcmb_muk**2

        # Save each of the four spectra
        if config['lensing']:
            for s in ['pp']:
                block[cmb_cl, s] = np.nan_to_num(
                    c_ell_data[s][2:]/2/np.pi*ell*(ell+1.0))

        for s in ['tt', 'ee', 'te', 'bb']:
            block[cmb_cl, s] = np.nan_to_num(c_ell_data[s][2:] * f)


def execute(block, config):
    import classy
    c = config['cosmo']

    try:
        # Set input parameters
        params = get_class_inputs(block, config)
        c.set(params)

        # Run calculations
        c.compute()

        # Extract outputs
        get_class_outputs(block, c, config, params)
    except classy.CosmoError as error:
        if config['debug']:
            sys.stderr.write("Error in class. You set debug=T so here is "
                             "more debug info:\n")
            traceback.print_exc(file=sys.stderr)
        else:
            sys.stderr.write("Error in class. Set debug=T for info: "
                             "{}\n".format(error))
        # HI_CLASS_NEW: if save_errors add failures to save_errors
        if config['save_errors']:
            fe = open(config['save_errors_file'], 'a')
            fe.write('{}\n'.format(
                '\t'.join([str(block[x]) for x in config['varying_params']])))
            fe.close()
        return 1
    finally:
        # Reset for re-use next time
        c.struct_cleanup()
    return 0


def cleanup(config):
    config['cosmo'].empty()


def get_arrays_class(block, name):
    array = []
    count = 1
    while block.has_value(cosmo, name+'__'+str(count)):
        array.append(block[cosmo, name+'__'+str(count)])
        count += 1
    str_array = ", ".join(map(str, array))
    return str_array


def try_to_get_arrays_class(params, block, name, name_save=None):
    if not name_save:
        name_save = name
    value = get_arrays_class(block, name)
    if value:
        params[name_save] = value
    return


def try_to_change_name(params, name_old, name_new):
    try:
        params[name_new] = params[name_old]
        params.pop(name_old)
    except KeyError:
        pass
    return


def get_varying_params(ini_values):
    """
    Get varying parameters (all, not just cosmological).
    Varying parameters are those which values are arrays of len 3.
    """
    list_params = []
    for (sec, key), val in ini_values:
        val = val.split()
        if len(val) == 3:
            list_params.append((sec, key))
    return list_params


def floatify_array(string_array):
    array = string_array.split(",")
    array = [float(x) for x in array]
    return np.array(array)


def stringify_array(array):
    string_array = [str(x) for x in array]
    string_array = ", ".join(string_array)
    return string_array


def tanh_from_derivs(x_nodes, y0, yp_nodes):
    y_nodes = np.ones_like(x_nodes)
    # First node is maximum z
    y_nodes[0] = y0
    for num, _ in enumerate(yp_nodes):
        # Derivatives are w.r.t. z
        y_nodes[num+1] = y_nodes[num] + yp_nodes[num]*(x_nodes[num+1] - x_nodes[num])
    return y_nodes
