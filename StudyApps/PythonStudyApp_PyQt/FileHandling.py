import os


def check_directory(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def make_trial_sequence(sub_num: int):
    import random
    import itertools
    name = lambda t, e, b, c, s: 'T' + str(t) + '_E' + str(e) + '_B' + str(b) + '_C' + str(c) + '_S' + str(s)
    # output = [('#SUB'+str(sub_num))]
    output=[]
    targets = [0, 1, 2, 3, 4, 5, 6, 7]
    environments = ['U', 'W']
    repetition = 5
    if sub_num % 2 == 0:
        environments.reverse()
    for i, env in enumerate(environments):
        for block in range(repetition):
            random.shuffle(targets)
            for c, target in enumerate(targets):
                output.append(name(target, env, block, c, sub_num))
        if i == len(environments) - 1:
            output.append("FINISH")
        else:
            output.append('BREAK')
    return output
