'''
Quick and dirty code to run a sweep over a set of lattice hyperparameters
'''
import subprocess
import itertools

choices = {
    'lattice_metric': ['match1'],
    'mean_override': [None],
    'd_length': [None],
    'count_aware': [False],
    'match_uniform': [True],
    'total_length': list(range(7,40)),
    'deviation': list(range(0,10)),
}

def make_cmd(lattice_metric, mean_override=None, d_length=None, count_aware=None, match_uniform=None, total_length=None, deviation=0):
    cmd = ["python3 src/mbr_rouge.py"]
    cmd.append(f'--lattice_metric {lattice_metric}')
    if total_length:
        cmd.append(f'--target_length {total_length}')
    if mean_override:
        cmd.append(f'--mean_override {mean_override}')
    if d_length is not None:
        cmd.append(f'--d_length {d_length}')
    if count_aware:
        cmd.append(f'--count_aware')
    if match_uniform:
        cmd.append(f'--match_uniform')
    cmd.append(f'--length_deviation {deviation}')
    return ' '.join(cmd)

keys = choices.keys()
values = [choices[k] for k in keys]
def main():
    for config in itertools.product(*values):
        #if config[-2] == 1 and config[-1] != 1:
        #    continue
        cmd = make_cmd(*config)
        subprocess.run(cmd.split(' '))
    
if __name__ == '__main__':
    main()
