# -*- coding: utf-8 -*-
"""

jpatrickhall@gmail.com
11.8.14

Simple script to update stemlist.

Assumes:
    update_stemlist.py (this file)
    replacements.txt (old replacement file)
    add.txt
    change.txt
    delete.txt

are in the same dir.

Assumes words to be replaced are unique keys.
Stems to replace keys can repeat.

Generates new replacement file: replacements_.txt.

Educational use only.

"""

import re

def load_dict(in_file):
    """ Loads a dictionary from a two column text file.

    Two columns must be separated by a whitespace or a comma.

    Arg:
        in_file: File containing the dictionary keys and values in two
            separate columns. First column will be keys. Second column will be
            values.

    Returns:
        A dictionary made from the in_file.

    """

    dict_ = {}
    with open(in_file) as file_:
        for line in file_:
            if not re.split(r'\s|,', line)[0].startswith('#') or \
                re.split(r'\s|,', line)[0] in ['', '\r\n', '\n']:
                pair = re.split(r'\s|,', line)
                key_ = pair[0].strip().lower()
                value_ = pair[1].strip().lower()
                dict_[key_] = value_
                
    return dict_

REPLACE_DICT = load_dict('C:\\Temp\\replacements.txt')
print 'Original stem dictionary contains %i terms.' % (len(REPLACE_DICT))

ADD_DICT = load_dict('add.txt')
print 'Add dictionary contains %i terms.' % (len(ADD_DICT))

CHANGE_DICT = load_dict('change.txt')
print 'Change dictionary contains %i terms.' % (len(CHANGE_DICT))

DEL_DICT = load_dict('delete.txt')
print 'Delete dictionary contains %i terms.' % (len(DEL_DICT))

### Add terms
REPLACE_DICT.update(ADD_DICT)

### Change terms
for key in CHANGE_DICT:
    REPLACE_DICT[key] = CHANGE_DICT[key]

### Delete terms
for key in DEL_DICT:
    if key in REPLACE_DICT:
        del REPLACE_DICT[key]

print 'Updated stem dictionary contains %i terms.' % (len(REPLACE_DICT))

### Output replacements_.txt
with open('replacements_.txt', 'wb') as o:
    o.write('### word lemma\n')
    for k, v in sorted(REPLACE_DICT.items()):
        o.write(str(k))
        o.write(' ')
        o.write(str(v))
        o.write('\n')

