import camb_vs_class_tools as tools

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Call the parser
    args = tools.argument_parser()

    # Initialize two codes to be compared
    code_ref = tools.Code(args.params_ref)
    code_ref.run()
    code_ref.load_data()
    if args.verbose:
        code_ref.print_keys()

    code_2nd = tools.Code(args.params_2nd)
    code_2nd.run()
    code_2nd.load_data()
    if args.verbose:
        code_2nd.print_keys()

    diff = code_ref.diff_codes(code_2nd)
    if args.verbose:
        diff.print_keys()

    only_ref = code_ref.only_here(code_2nd)
    if args.verbose:
        only_ref.print_keys()
