'''
Quick and dirty code to run a sweep over a set of lattice hyperparameters
'''
import subprocess
import itertools

choices = {
    'lattice_metric': ['match2', 'rouge2'],
    'mean_override': [None, 20, 25, 30, 35],
    'd_length': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'count_aware': [False],
    'match_uniform': [True],
    #'target_length': list(range(7,30)),
    #'deviation': list(range(0,10)),
}
# uniform
# length_alpha
# 


def make_cmd(lattice_metric, mean_override, d_length, count_aware, match_uniform, target_length, deviation):
    cmd = ["python3 src/mbr_rouge.py --rerank_topk 50"]
    cmd.append(f'--lattice_metric {lattice_metric}')
    if mean_override:
        cmd.append(f'--mean_override {mean_override}')
    if d_length is not None:
        cmd.append(f'--d_length {d_length}')
    if count_aware:
        cmd.append(f'--count_aware')
    if match_uniform:
        cmd.append(f'--match_uniform')
    cmd.append(f'--target_length {target_length}')
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
