from os import listdir, rename
from os.path import isfile, join

org_dir = '/home/behzad/Data/Palmetto/oyster/makesense/train/'
# org_dir = '/home/behzad/Data/Palmetto/oyster/makesense/val/'
org_files = [f.split('.')[0] for f in listdir(org_dir)
             if isfile(join(org_dir, f)) and f.split('.')[-1] == 'json']
new_files = [f.split('_')[1] for f in org_files]
new_files = [f[1:] if f[0] == '0' else f for f in new_files]
# new_files = ['2'+f[1:] if f[0] == '0' else f for f in new_files]
for ext in ['.json', '.jpg']:
    for (of, nf) in zip(org_files, new_files):
        rename(join(org_dir, of+ext), join(org_dir, nf+ext))
