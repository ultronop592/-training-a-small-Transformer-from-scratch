import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file, e.g. config/train_inabs.py
        assert not arg.startswith('--')
        config_file = arg
        print(f"overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]

        if key in globals():
            try:
                # attempt to evaluate it (e.g. for int, float, bool, None, etc.)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string value
                attempt = val

            # ensure the types match (prevents accidentally setting a string
            # where an int was expected, etc.)
            assert type(attempt) == type(globals()[key])

            print(f"overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"unknown config key: {key}")