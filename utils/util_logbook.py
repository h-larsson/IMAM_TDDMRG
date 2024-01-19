import pickle


def parse(inputs0):
    inputs = inputs0.copy()

    for key in inputs:
        if inputs[key] == 'logbook':
            assert inputs['prev_logbook'] is not None, f'The input \'{key}\' is using ' + \
                'the value of the same input from a logbook but no logbook location is ' + \
                'specified in the input file. Provide the path to a logbook using ' + \
                '\'prev_logbook\' input keyword.'
            with open(inputs['prev_logbook'], 'rb') as f:
                logbook = pickle.load(f)
            break

    for key in inputs:
        if inputs[key] == 'logbook':
            assert key in logbook, f'The input \'{key}\' cannot be found in the ' + \
                'specified previous logbook. Check your previous logbook again.'
            inputs[key] = logbook[key]

    return inputs
