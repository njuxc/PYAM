# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 15:45:41 2021

@author: 11834
"""

from github import Github
import sys
from tqdm import tqdm
import binascii
from datetime import datetime

from git import Repo, Commit

repo = Repo("pandas/")
commits = list(repo.iter_commits('master'))

# g = Github("80fafa7d02c7e6219badcd9bbb8034991a0cbfb5")
# repo = g.get_repo("pandas-dev/pandas")
# print("Getting commits...")
# commits = repo.get_commits()
print("Finish get commits.")

keys=['bug','Bug','fix','Fix','error','Error','ERROR','check','Check',
      'wrong','Wrong','nan','NAN','inf','issue','ISSUE','Issue','fault','Fault',
      'fail','Fail','FAIL','carsh','Crash']

api_keys=['API','api','Api','missing check','null point','return','parameter',
          'arg','para','ARG']


txt=''
atxt=''
print("All commits: " + str(len(commits)))
for commit in tqdm(commits):
    # files=list(map(lambda c: c.filename, commit.files))
    # passed=False
    # for f in files:
    #     if f.endswith('.py'):
    #         passed=True
    # if not passed:
    #     continue

    # print(commit.stats.files)

    mess=commit.message
    if 'typo' in mess:
        continue
    for k in keys:
        if k in mess:            
            txt+='**************************************************\n'
            txt+='commit id:'+ binascii.b2a_hex(commit.binsha).decode("utf-8")+'\n'
            txt+='commit date:' + str(datetime.fromtimestamp(commit.committed_date))+'\n'
            # txt+='commit url:' + str(commit.html_url)+'\n'
            txt+='commit message:' + str(commit.message)+'\n'
            txt+='commit files:'+str(commit.stats.files)+'\n'
            txt+='**************************************************\n'
    with open('pandas.txt','w+',encoding='utf-8') as f:
        f.write(txt)

        
    for kj in api_keys:
        if kj in mess:
            
           atxt+='**************************************************\n'
           atxt+='commit id:'+ binascii.b2a_hex(commit.binsha).decode("utf-8")+'\n'
           atxt+='commit date:' + str(datetime.fromtimestamp(commit.committed_date))+'\n'
        #    atxt+='commit url:' + str(commit.html_url)+'\n'
           atxt+='commit message:' + str(commit.message)+'\n'
           atxt+='commit files:'+str(commit.stats.files)+'\n'
           atxt+='**************************************************\n'
           with open('pandas_api.txt','w+',encoding='utf-8') as f:
               f.write(atxt)
    
    