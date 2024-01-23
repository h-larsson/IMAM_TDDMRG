import pickle


def parse(inputs0):
    inputs = inputs0.copy()

    for key in inputs:
        if inputs[key] == 'logbook':
            assert inputs['prev_logbook'] is not None, f"The input '{key}' is using " + \
                'the value of the same input from a logbook but no logbook location is ' + \
                'specified in the input file. Provide the path to a logbook using ' + \
                '\'prev_logbook\' input keyword.'
            with open(inputs['prev_logbook'], 'rb') as f:
                logbook = pickle.load(f)
            break

    for key in inputs:
        if isinstance(inputs[key], str):
            key0 = None
            if inputs[key] == 'logbook':
                key0 = key
            elif inputs[key][0:8] == 'logbook:':
                key0 = inputs[key][8:]
            else:
                pass

            if key0 is not None:
                assert key0 in logbook, f"The keyword '{key0}' cannot be found in the " + \
                    'specified previous logbook. Check your previous logbook again.'
                inputs[key] = logbook[key0]
        else:
            pass
    return inputs
