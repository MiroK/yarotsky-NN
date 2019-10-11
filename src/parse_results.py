import numpy as np
import re


def extract_data(path):
    '''We want to get yarotsky, min/max/mean/std of learned errors'''
    number = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')

    with open(path, 'r') as f:
        dofs = int(number.findall(next(f))[0])
        data = np.loadtxt(path)

        eY, eL = data[0], data[1:]

    return eY, np.min(eL), np.max(eL), np.mean(eL), np.std(eL), dofs

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse, glob, os

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-architecture', type=str, default='share', choices=['share', 'noshare', 'both'])
    parser.add_argument('-root', type=str, default='./results')
    args = parser.parse_args()

    number = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')

    archs = []
    if args.architecture == 'both':
        archs = ['share', 'noshare']
    else:
        archs = [args.architecture]
    colors = ['blue', 'red']
        

    # Absolute
    fig, ax = plt.subplots()
    ax_right = ax.twinx()
    
    ax.set_yscale('log', basey=2)
    ax.set_xlabel('$m$')
    ax.set_ylabel('$||x^2 - f_m||_{\infty}$')
    
    ax_right.set_ylabel('dofs(f_m)')

    left, right = [], []
    for color, arch in zip(colors, archs):
        # Get all candidates
        data_set = glob.glob(os.path.join(args.root, '%s_*' % arch))
        assert data_set

        ms = map(int, [number.findall(d)[0] for d in data_set])
        ms, data_set = zip(*sorted(zip(ms, data_set), key=lambda p: p[0]))
    
        data = [extract_data(d) for d in data_set]
        eYs, mins, maxs, means, stds, dofss = np.array(data).T

        if not left:
            l = ax.plot(ms, eYs, marker='x', color='green')[0]
            left.append((l, 'Yarotsky'))
        
        l = ax.plot(ms, mins, marker='o', color=color)[0]
        left.append((l, '%s:min(Learned)' % arch))
        
        l = ax.errorbar(ms, means, yerr=stds, marker='d', color=color)[0]
        left.append((l, '%s:mean(Learned)' % arch))

        l = ax_right.plot(ms, dofss, marker='s', color=color, linestyle='dashed')[0]
        right.append((l, arch))
        
    ax.legend(*zip(*left), loc='center left')
    ax_right.legend(*zip(*right), loc='center right')

    # Relative to Yarotsky
        # Absolute
    fig, ax = plt.subplots()

    ax.set_yscale('log', basey=2)
    ax.set_xlabel('$m$')
    ax.set_ylabel('$error(Learned_m)/error(Yarotsky_m)$')
    
    ax_right.set_ylabel('dofs(f_m)')
    left = []
    for color, arch in zip(colors, archs):
        # Get all candidates
        data_set = glob.glob(os.path.join(args.root, '%s_*' % arch))
        assert data_set

        ms = map(int, [number.findall(d)[0] for d in data_set])
        ms, data_set = zip(*sorted(zip(ms, data_set), key=lambda p: p[0]))
    
        data = [extract_data(d) for d in data_set]
        eYs, mins, maxs, means, stds, dofss = np.array(data).T

        l = ax.plot(ms, mins/eYs, marker='o', color=color)[0]
        left.append((l, '%s:min' % arch))
        
        l = ax.plot(ms, means/eYs, marker='d', color=color)[0]
        left.append((l, '%s:mean' % arch))

    ax.legend(*zip(*left), loc='lower right')

    plt.show()
