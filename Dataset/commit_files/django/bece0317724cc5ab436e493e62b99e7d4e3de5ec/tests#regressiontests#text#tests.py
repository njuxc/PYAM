# coding: utf-8
r"""
# Tests for stuff in django.utils.text and other text munging util functions.

>>> from django.utils.text import *

### smart_split ###########################################################
>>> list(smart_split(r'''This is "a person" test.'''))
[u'This', u'is', u'"a person"', u'test.']
>>> print list(smart_split(r'''This is "a person's" test.'''))[2]
"a person's"
>>> print list(smart_split(r'''This is "a person\"s" test.'''))[2]
"a person"s"
>>> list(smart_split('''"a 'one'''))
[u'"a', u"'one"]
>>> print list(smart_split(r'''all friends' tests'''))[1]
friends'

### urlquote #############################################################
>>> from django.utils.http import urlquote, urlquote_plus
>>> urlquote(u'Paris & Orl\xe9ans')
u'Paris%20%26%20Orl%C3%A9ans'
>>> urlquote_plus(u'Paris & Orl\xe9ans')
u'Paris+%26+Orl%C3%A9ans'

### iri_to_uri ###########################################################
>>> from django.utils.encoding import iri_to_uri
>>> iri_to_uri(u'red%09ros\xe9#red')
'red%09ros%C3%A9#red'
>>> iri_to_uri(u'/blog/for/J\xfcrgen M\xfcnster/')
'/blog/for/J%C3%BCrgen%20M%C3%BCnster/'
>>> iri_to_uri(u'locations/%s' % urlquote_plus(u'Paris & Orl\xe9ans'))
'locations/Paris+%26+Orl%C3%A9ans'

iri_to_uri() is idempotent:
>>> iri_to_uri(iri_to_uri(u'red%09ros\xe9#red'))
'red%09ros%C3%A9#red'
"""
