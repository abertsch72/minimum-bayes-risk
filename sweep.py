'''
Quick and dirty code to run a sweep over a set of lattice hyperparameters
'''
import subprocess
import itertools

choices = {
    'lattice_metric': ['rouge1', 'rouge2', 'match1', 'match2'],
    'match_uniform': [True, False],
    'd_length': [4, None],
    'lattice_topk': [1, 5, 10, 30],
    'rerank_rouge': [1, 2, 6]
}

def make_cmd(lattice_metric, match_uniform, d_length, lattice_topk, rerank_rouge):
    cmd = ["python3 src/mbr_rouge.py"]
    cmd.append(f'--lattice_metric {lattice_metric}')
    if match_uniform:
        cmd.append('--match_uniform')
    if d_length is not None:
        cmd.append(f'--d_length {d_length}')
    cmd.append(f'--lattice_topk {lattice_topk}')
    cmd.append(f'--rerank_rouge {rerank_rouge}')
    return ' '.join(cmd)

keys = choices.keys()
values = [choices[k] for k in keys]
def main():
    for config in itertools.product(*values):
        if config[-2] == 1 and config[-1] != 1:
            continue
        cmd = make_cmd(*config)
        subprocess.run(cmd.split(' '))
    
if __name__ == '__main__':
    main()