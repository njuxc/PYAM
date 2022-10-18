import re
import os
import tokenize
from tqdm import tqdm
import json
from typing import List
from gensim.models import Word2Vec, KeyedVectors



def remove_comments_and_docstrings(source: str):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            # note: a space and not an empty string
            return ' '
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != '':
            temp.append(x)
    return '\n'.join(temp)

# ------------Tokenizers--------------

def tokenize_with_camel_case(token) :
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
    return [m.group(0) for m in matches]

def tokenize_with_snake_case(token):
    return token.split('_')

def get_subtokens(token):
    snake_case_tokenized = []

    snake_case_tokenized.extend(tokenize_with_snake_case(token))

    camel_case_tokenized = []
    for token in snake_case_tokenized:
        camel_case_tokenized.extend(tokenize_with_camel_case(token))
        
    return camel_case_tokenized

def get_tokens(filename) -> List:
    with open(filename, 'rb') as f:
        tokens = tokenize.tokenize(f.readline)
        try:
            return [token for token in tokens]
        except Exception as e:
            print(e)
            print("Error filename: " + filename)

# tokens = get_tokens("sample_python.py")
# for token in tokens:
#     print(token)

# -------------Vocabulary------------

def build_vocabulary_codetext(filenames, vector_size, uncase):
    subtokens = []
    for filename in tqdm(filenames, desc='Remove utf-8 undecodable lines'):
        f = open(filename, 'rb')
        lines = []
        original_lines = f.read().splitlines()
        for line in original_lines:
            ll = None
            try:
                ll = line.decode('utf-8')
            except Exception as e:
                pass
            if ll != None:
                lines.append(ll)
        f.close()
        if len(lines) != len(original_lines):
            # print("Rewrite file: " + filename)
            f = open(filename, 'w', encoding='utf-8')
            for line in lines:
                f.write(line + '\n')
            f.close()

    for filename in tqdm(filenames, desc='Splitting py files into subtokens'):
        tokens = get_tokens(filename)
        line = []
        for token_type, token_val, _, _, line_str in tokens:
            if (token_type == tokenize.NUMBER):
                line.append('<number>')
            elif (token_type == tokenize.STRING):
                line.append('<string>')
            else:
                line.extend([s.lower() if uncase else s for s in get_subtokens(token_val)])
        subtokens.append(line)
    w2v = Word2Vec(subtokens, 
                   vector_size=vector_size, 
                   window=5, 
                   max_vocab_size=10000, 
                   min_count=1, 
                   workers=32)
    return w2v

def build_vocabulary_astnodes(asts, vector_size):
    all_tokens = []
    for ast in tqdm(asts, desc='Training w2v model for ast nodes'):
        all_tokens.append([node['type'] for node in ast])
    w2v = Word2Vec(all_tokens,
                   vector_size=vector_size,
                   window=5,
                   max_vocab_size=10000,
                   min_count=1,
                   workers=32)
    return w2v
        
def train_both_w2v(astfile='../ast_py150/python100k_train.json', 
                   src_file_descript='../src_py150/python100k_train.txt',
                   astnodes_vocab_size=256,
                   codetext_vocab_size=256):
    # First parse ast
    if not os.path.exists('w2v_astnodes.bin'):
        with open(astfile, 'r') as f:
            asts = []
            for line in f.read().splitlines():
                asts.append(json.loads(line))
            w2v_astnodes = build_vocabulary_astnodes(asts, astnodes_vocab_size).wv
            w2v_astnodes.save('w2v_astnodes.bin')
            print('Finish build astnodes w2v model.')
    
    if not os.path.exists('w2v_codetext.bin'):
        with open(src_file_descript, 'r', encoding='utf-8') as f:
            path_prefix = '../src_py150/'
            filenames = []
            for filepath in f.read().splitlines():
                filepath = path_prefix + filepath
                filenames.append(filepath)
            w2v_codetext = build_vocabulary_codetext(filenames, codetext_vocab_size, True).wv
            w2v_codetext.save('w2v_codetext.bin')
            print('Finish build codetext w2v model.')

# train_both_w2v()
