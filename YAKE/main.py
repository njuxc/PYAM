from rake_nltk import Rake
from git import Repo
from tqdm import tqdm
from pke.unsupervised import YAKE
from nltk.corpus import stopwords
import json
import binascii
import os

# 分词，停词
# Candidate Generation
# Candidate Scoring
# Post-processing
# Ranking

# 可以调节的参数：
n_gram_size = 4  # 关键词提取的窗口大小
n_ranked_candidates = 20  # 最后生成候选的数量
threshold = 0.6

repo_list_dict = {
    'airflow': 'main',
    'ansible': 'devel',
    'django': 'main',
    'keras': 'master',
    'pandas': 'main',
    'scikit-learn': 'main'
}


def gen_commit_msg():
    f = open('commit_message.json', 'w')
    cmt_msg_dict = {}

    for repo_name, branch_name in repo_list_dict.items():
        print("Parsing " + repo_name + ": ")
        repo = Repo("repo/" + repo_name + "/")
        commits = list(repo.iter_commits(branch_name))
        for commit in tqdm(commits):
            k = repo_name + '#' + binascii.b2a_hex(commit.binsha).decode("utf-8")
            cmt_msg_dict[k] = str(commit.message)

    json.dump(cmt_msg_dict, f)
    f.close()


def load_commit_category() -> dict:
    commit_category_dict = {}
    repo_commit_list = open("commit_ids.txt", 'r').read().strip().split("end")
    repo_category_list = open("category.txt", 'r').read().strip().split("END")
    for single_repo_cmlist, single_repo_catelist in zip(repo_commit_list, repo_category_list):
        cmlist_lines = single_repo_cmlist.strip().splitlines()
        catelist_lines = single_repo_catelist.strip().splitlines()
        repo_name = cmlist_lines[0]
        commit_id_list = cmlist_lines[1:]
        category_list = catelist_lines[1:]
        assert len(commit_id_list) == len(category_list)

        for i in range(len(commit_id_list)):
            key = repo_name + "#" + commit_id_list[i]
            if key in commit_category_dict.keys():
                commit_category_dict[key].append(category_list[i])
            else:
                commit_category_dict[key] = [category_list[i]]
    # print(json.dumps(commit_category_dict, indent=2))
    return commit_category_dict


if __name__ == '__main__':
    if not os.path.exists('commit_message.json'):
        gen_commit_msg()

    commit_category = load_commit_category()
    cmt_msg_dict = json.load(open("commit_message.json", 'r', encoding='utf-8'))

    categories_doc_dict = {
        "unexpected behavior or result": [],
        "exception and error": [],
        "crash": [],
        "redundant consumption": [],
        "security hole": [],
        "warning": [],
        "irregularity": [],
        "incompatibility": []
    }

    # rake = Rake()
    extractor = YAKE()
    for key, category_list in commit_category.items():
        try:
            cmt_msg = cmt_msg_dict[key]
        except KeyError as e:
            print(key)
            continue

        for category in category_list:
            try:
                categories_doc_dict[category].append(cmt_msg)
            except KeyError as e:
                print("Category not exist: " + category)

    for category, doc_list in categories_doc_dict.items():
        extractor.load_document("\n".join(doc_list), "en", stoplist=stopwords.words('english'))
        extractor.candidate_selection(n=n_gram_size)
        extractor.candidate_weighting(window=n_gram_size, use_stems=False)

        key_phrases = extractor.get_n_best(n=n_ranked_candidates, threshold=threshold, stemming=False)
        print("=" * 20)
        print("CATEGORY " + category + ":")
        print(key_phrases)

    # rake.extract_keywords_from_text(cmt_msg)
    # print(rake.get_ranked_phrases_with_scores())
