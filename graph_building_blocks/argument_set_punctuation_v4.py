'''
Argument word set version 4.0.

Date: 25/10/2020
Author: Yinya Huang.
'''

'''
* Argument word set is extracted from PDTB 2.0, relation = "Explicit".
    - In this version, try assigning all argument words with pattern=4, i.e., return (h, r, t) & (t, r, h).
    - Further considering relation patterns as in v2 & v1 in version 4.1.

* Punctuation set stays the same with version 2.0.

Save relations to json file. 

'''

import json
import sys
sys.path.append('../')
from tqdm import tqdm
from pdtb_toolkit.pdtb import CorpusReader



punctuations = [
    ',', '.', ';', ':',
    '<s>', '</s>'
    ]




if __name__ == '__main__':

    output_dir = 'arg_set_v4.json'

    pdtb_dir = '/Users/mac/Documents/__datasets/PDTB2/pdtb2.csv'
    pdtb = CorpusReader(pdtb_dir)

    explicit_dict, implicit_dict, altlex_dict, entrel_dict, norel_dict = {}, {}, {}, {}, {}
    save_argument_set = {}
    for datum in tqdm(pdtb.iter_data()):

        relation = datum.Relation
        connective_rawtext = datum.Connective_RawText
        connhead = datum.ConnHead
        filenumber = datum.FileNumber

        key = '{} | {}'.format(connective_rawtext, connhead)

        if relation == 'Explicit':
            if not key in explicit_dict:
                explicit_dict[key] = []
            explicit_dict[key].append(filenumber)

            if not connhead in save_argument_set:
                save_argument_set[connhead] = 4
            # save_argument_set[connhead].append(connective_rawtext)
    # save_argument_set = [ k,list(set(v)) for k,v in save_argument_set.items()]
    # for k,v in save_argument_set.items():
    #     save_argument_set.update({k:list(set(v))})


    with open('explicit_rels.json', 'w') as f:
        json.dump(explicit_dict, f)
    with open('explicit_' + output_dir, 'w') as f:
        json.dump(save_argument_set, f)