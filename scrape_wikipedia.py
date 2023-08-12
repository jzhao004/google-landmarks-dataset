import os

import numpy as np
import pandas as pd

import mwclient
from nltk import pos_tag, word_tokenize
from tqdm import tqdm 

from wikipedia import *


ann_root = 'data/annotations'
label_to_category_pth = os.path.join(ann_root, 'train_label_to_category.csv')
train_pth = os.path.join(ann_root, 'train_clean.csv')
save_to = os.path.join(ann_root, 'train_clean_updated.csv')

site = mwclient.Site('commons.wikimedia.org')
wikipedia = Wikipedia()

omit_sec = {'See also', 'Sources', 'Further reading', 'Footnotes', 'Bibliography', 'External links'} # omit sections 
omit_sec_contain = {'gallery', 'notes', 'references'} # omit sections with titles containing
incl_pos = ['NN', 'NNS', 'NNP', 'NNPS'] # nouns and pronouns

n_characteristics = 3 # max number of characteristics per sample

def remove_bracketed_words(text): 
    # get indexes of open and close brackets 
    oidx, cidx = [], []
    for i, ch in enumerate(text): 
        if ch == '(':
            oidx.append(i)
        elif ch == ')':
            cidx.append(i)
    
    # get placement of open and close brackets
    idx = np.array(oidx + cidx)
    ch = np.array(['o']*len(oidx) + ['c']*len(cidx))
    
    sort = np.argsort(idx)
    idx = idx[sort].tolist()
    ch = ch[sort].tolist()
    
    # find starting and ending indexes of brackets
    # do not include brackets within brackets
    count = 0
    ss, ee = -1, -1
    bracketed_words = []
    
    for idx_, ch_ in zip(idx, ch): 
        if ch_ == 'o':
            count += 1 
            ss = idx_ if (ss == -1) else ss
                
        elif ch_ == 'c':
            count -= 1 
        
        if count == 0: 
            ee = idx_
            
            # include leading whitespace if any
            if ss != 0: 
                word = text[ss-1:ee+1]
                if word[0] != ' ': 
                    word = word[1]
            else:
                word = text[ss:ee+1]
                
            bracketed_words.append(word)
            
            ss, ee = -1, -1 
    
    for b in bracketed_words: 
        text = text.replace(b, '')
        
    return text

if __name__ == "__main__":
    label_to_category = pd.read_csv(label_to_category_pth)
    train = pd.read_csv(train_pth)
    train = train.merge(label_to_category, on='landmark_id')

    print(f'No. of training samples: {len(train)}')
    print(f'No. of categories: {len(label_to_category)}')

    train['page_title'] = None
    train['lat'] = None
    train['lon'] = None
    train['characteristics'] = None

    for i in tqdm(range(len(train))):
        # get category title from wikimedia
        category_url = train.loc[i, 'category']

        try:
            page = site.pages[os.path.split(category_url)[-1]]
        except:
            continue

        # search wikipedia
        try:
            page_title = wikipedia.search(page.page_title)[0]
        except: 
            continue 

        wiki_page = wikipedia.page(page_title)

        # get page title and location coordinates
        train.loc[i, 'page_title'] = page_title
        train.loc[i, 'lat'], train.loc[i, 'lon'] = wikipedia.coordinates(page_title)

        # get first 2 sentences of wikipedia summary
        train.loc[i, 'summary'] = wiki_page.summary(exsentences=2)

        # get section titles 
        ## exclude sections in omit_sec and sections with titles containing words in omit_sec_contain
        ## include only sections with titles starting and ending with nouns/pronouns 
        section_titles = set(wiki_page.section_titles)
        section_titles = section_titles - section_titles.intersection(omit_sec)
        section_titles = list(s for s in section_titles if not any([k in s.lower() for k in omit_sec_contain]))

        final_section_titles = [] 
        for s in section_titles: 
            pos = pos_tag(word_tokenize(s))
            if (pos[0][-1] in incl_pos) & (pos[-1][-1] in incl_pos):
                final_section_titles.append(s)

        if len(final_section_titles) > 0: 
            characteristics = {}
            count = 0 

            # get first 2 sentences of section description 
            for section_title in final_section_titles:
                section = wiki_page.section_by_title(section_title, exsentences=2) 
                if section['text']:
                    section_text = remove_bracketed_words(section['text'])
                    characteristics[section_title] = section_text
                    count += 1

                if count >= n_characteristics: 
                    break

            train.loc[i, 'characteristics'] = str(characteristics)

        if i%10 == 0:
            train.to_csv(save_to, index=False)

    # remove samples with missing information            
    train = train.dropna().reset_index(drop=True)
    train.to_csv(save_to, index=False)
