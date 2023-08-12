# Adapted from https://github.com/goldsmith/Wikipedia and https://github.com/martin-majlis/Wikipedia-API

import requests

from collections import defaultdict
import re
from typing import Dict, List

import nltk
#nltk.download('popular')


class Wikipedia:
    def __init__(self):
        """
        Constructs Wikipedia object for extracting information from Wikipedia
        """

        self.api_url = 'http://en.wikipedia.org/w/api.php'
        self.language = 'en'
        self.ns = 0 # Namespace: MAIN

        self._session = requests.Session()

        default_headers = dict()
        default_headers.setdefault(
            'User-Agent',
            'Wikipedia-API',
        )
        self._session.headers.update(default_headers)

    def __del__(self):
        """ Closes session """
        self._session.close()

    def search(self, query, limit=1):
        """
        Search for relevant pages given query

        Args:
            query (str): search query
            limit (int, optional): maxmimum number of results to return (default is 1)

        Returns:
            (List[str]): page titles
        """

        params = {
        'action': 'query',
        'list': 'search',
        'format':'json',
        'srsearch': query,
        'srlimit': limit, 
        }

        res = self._query(params)

        if 'error' in res:
            raise Exception(res['error']['info'])

        res = [r['title'] for r in res['query']['search']]

        return res

    def page(self, title):
        """
        Creates page with given title
        
        Args:
            title (str): page title
        """
        # search Wikipedia for relevant pages
        page = WikipediaPage(self, title=title)

        if not page.exists():
            raise Exception('Wikipedia page does not exists')
    
        return page

    def extracts(self, page):
        """
        Returns summary of given page 
        
        Args:
            page (WikipediaPage object)
        """
        params = {
            'action': 'query',
            'format': 'json',
            'titles': page.title,
            'prop': 'extracts',
            'explaintext': 1,
            'exsectionformat': 'wiki',
            'redirects': 1
        }
        res = self._query(params)
        pages = res['query']['pages']

        for k, v in pages.items():
            k = int(k)
            page._attributes['pageid'] = k

            if k == -1:
                return ''

            return self._build_extracts(v, page)

    def coordinates(self, title):
        params = {
            'action': 'query',
            'format':'json',
            'titles': title,
            'prop':'coordinates'            
        }

        res = self._query(params)

        if 'error' in res:
            raise Exception(res['error']['info'])
        
        try:
            pages = res['query']['pages']
            for key in pages:
                coordinates = pages[key]['coordinates'][0]
                return coordinates['lat'], coordinates['lon']
        except:
            return None, None
           
    def _query(self, params):
        """ 
        Queries Wikipedia API to fetch content 
        
        Args: 
            params (Dict): parameters used in API call
        
        """

        r = self._session.get(self.api_url, params=params, timeout=10.0)
        
        return r.json()

    def _build_extracts(self, extract, page):
        """ 
        Constructs summary of given page

        Args:
            extract (Dict)
            page (WikipediaPage object)
        
        """
        page._summary = ''
        page._section_mapping = defaultdict()

        section_stack = [page]
        section = None
        prev_pos = 0
        
        extract_format = re.compile(r'\n\n *(===*) (.*?) (===*) *\n')
        for match in re.finditer(extract_format, extract["extract"]):
            if len(page._section_mapping) == 0:
                page._summary = extract["extract"][0 : match.start()].strip()
            elif section is not None:
                section._text = (extract["extract"][prev_pos : match.start()]).strip()

            sec_title = match.group(2).strip()
            sec_level = len(match.group(1))
            section = self._create_section(sec_title, sec_level)
            sec_level = section.level + 1

            if sec_level > len(section_stack):
                section_stack.append(section)
            elif sec_level == len(section_stack):
                section_stack.pop()
                section_stack.append(section)
            else:
                for _ in range(len(section_stack)-sec_level+1):
                    section_stack.pop()
                section_stack.append(section)

            section_stack[len(section_stack)-2]._section.append(section)

            prev_pos = match.end()
            page._section_mapping[section.title] = section

        # pages without sections have only summary
        if page._summary == '':
            page._summary = extract['extract'].strip()

        if prev_pos > 0 and section is not None:
            section._text = extract['extract'][prev_pos:]
        return page._summary

    def _create_section(self, title, level):
        """ 
                Creates section with given title and level

        Args: 
            title (str): section title
            level (int): section level
        """
        return WikipediaPageSection(self, title, level-1)


class WikipediaPage:

    ATTRIBUTES_MAPPING = {
        'title' : [],
        'ns' : [], 
        'language': [],
        'pageid': ['extracts'],
    }

    def __init__(self, wiki, title):
        """
        Constructs WikipediaPage object to represent a page

        Args:
            wiki (Wikipedia object)
            title (str): page title 
        """
        self.wiki = wiki
        self._summary = ''  # type: str
        self._section = []  # type: List[WikipediaPageSection]
        self._section_mapping = {}  # type: Dict[str, List[WikipediaPageSection]]

        self._called = {
            'extracts': False,
            'info': False,
        }

        self._attributes = {
            'title': title,
            'ns': self.wiki.ns,
            'language': self.wiki.language,
        }

    def __getattr__(self, name):
        if name not in self.ATTRIBUTES_MAPPING:
            return self.__getattribute__(name)

        if name in self._attributes:
            return self._attributes[name]

        for call in self.ATTRIBUTES_MAPPING[name]:
            if not self._called[call]:
                self._fetch(call)
                return self._attributes[name]

    def _fetch(self, call):
        """ 
        Fetches data 
            call (str): attribute of object
        """
        getattr(self.wiki, call)(self)
        self._called[call] = True
        return self

    def exists(self):
        """
        Returns True if current page exists, otherwise False
        """
        return bool(self.pageid != -1)

    def summary(self, exsentences=None):
        """
        Returns summary of current page

        Args: 
            exsentences (int, optional): maximum number of sentences to return. Return all if None (default is None) 
        """
        if not self._called['extracts']:
            self._fetch('extracts')

        summary = self._summary

        if exsentences is not None: 
            sentences = nltk.sent_tokenize(summary)
            summary = ' '.join(sentences[:exsentences])

        return summary

    @property
    def section_titles(self):
        """ 
        Returns titles of sections of current page
        """
        if not self._called['extracts']:
            self._fetch('extracts')

        return [s.title for s in self._section]

    def section_by_title(self, title, exsentences=None):
        """
        Returns section of current page with given title
        
        Args: 
            title (str): section title
            exsentences (int, optional): maximum number of sentences to return. Return all if None (default is None) 
        """
        if not self._called['extracts']:
            self._fetch('extracts')

        section = self._section_mapping.get(title)

        if section is None:
            return {
                'title' : title, 
                'text' : '', 
                }
            
        text = section.text
        # Limit no. of sentences returned
        if exsentences is not None: 
            sentences = nltk.sent_tokenize(text)
            text = ' '.join(sentences[:exsentences])

        s_dict = {
            'title' : title,
            'text' : text, 
            'subsections' : {} 
            }

        for i, ss_title in enumerate(section.subsection_titles): 
            ss_dict = section.subsection_by_title(ss_title, exsentences=exsentences)
            s_dict['subsections'][i] = ss_dict

        return s_dict

    @property
    def text(self):
        """
        Returns text 
        """
        text = self.summary()
        if len(text) > 0:
            text += '\n\n'

        for s_titles in self.section_titles:
            text += self._section_mapping.get(s_titles).full_text(level=2)

        return text.strip()

    def __repr__(self):
        if any(self._called.values()):
            return f'{self.title} (id: {self.pageid}, ns: {self.ns})'

        return f'{self.title} (id: ??, ns: {self.ns})'


class WikipediaPageSection:
    def __init__(self, wiki, title, level=0, text=''):
        """ 
        Constructs WikipediaPageSection object to represent a section in the current page 
        
        Args:
            wiki (Wikipedia object)
            title (str): title of current section
            level (int, optional): indentation level of current section (default is 0)
            text (str, optional): text of current section (default is '')
        """
        self.wiki = wiki
        self._title = title
        self._level = level
        self._text = text
        self._section = []

    @property
    def title(self):
        """
        Returns title of current section
        """ 
        return self._title

    @property
    def level(self):
        """
        Returns indentation level of current section
        """
        return self._level

    @property
    def text(self):
        """
        Returns text of current section
        """
        return self._text

    @property
    def subsection_titles(self):
        """
        Returns titles of subsections of current section 
        """
        return [ss.title for ss in self._section]

    def subsection_by_title(self, title, exsentences=None):
        """
        Returns subsection of current section with given title
        
        Args: 
            title (str): subsection title
            exsentences (int, optional): maximum number of sentences to return. Return all if None (default is None) 
        """
        for subsection in self._section: 
            if subsection.title == title: 
                text = subsection.text
                # Limit no. of sentences returned
                if exsentences is not None: 
                    sentences = nltk.sent_tokenize(text)
                    text = ' '.join(sentences[:exsentences])

                return {
                    'title' : subsection.title, 
                    'text' : text
                    }

        return None

    def full_text(self, level=1):
        """
        Returns text of current section and all its subsections
        
        Args:
            level (int, optional): indentation level (default is 1)
        """
        text = f'{self.title}\n'
        text += self._text

        if len(self._text) > 0:
            text += '\n\n'

        for subsection in self._section:
            text += subsection.full_text(level+1)

        return text

    def __repr__(self):
        return 'Section: {} ({}):\n{}\nSubsections ({}):\n{}'.format(
            self._title,
            self._level,
            self._text,
            len(self._section),
            '\n'.join(map(repr, self._section)),
        )