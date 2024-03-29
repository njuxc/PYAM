# -*- coding: utf-8 -*-

# Unit tests for cache framework
# Uses whatever cache backend is set in the test settings file.
from __future__ import unicode_literals

import os
import re
import shutil
import tempfile
import threading
import time
import unittest
import warnings

from django.conf import settings
from django.core import management
from django.core.cache import cache, caches, CacheKeyWarning, InvalidCacheBackendError
from django.db import connection, router, transaction
from django.core.cache.utils import make_template_fragment_key
from django.http import HttpResponse, StreamingHttpResponse
from django.middleware.cache import (FetchFromCacheMiddleware,
    UpdateCacheMiddleware, CacheMiddleware)
from django.template import Template
from django.template.response import TemplateResponse
from django.test import TestCase, TransactionTestCase, RequestFactory
from django.test.utils import (override_settings, IgnoreDeprecationWarningsMixin,
    IgnorePendingDeprecationWarningsMixin)
from django.utils import six
from django.utils import timezone
from django.utils import translation
from django.utils.cache import (patch_vary_headers, get_cache_key,
    learn_cache_key, patch_cache_control, patch_response_headers)
from django.utils.encoding import force_text
from django.views.decorators.cache import cache_page

try:    # Use the same idiom as in cache backends
    from django.utils.six.moves import cPickle as pickle
except ImportError:
    import pickle

from .models import Poll, expensive_calculation


# functions/classes for complex data type tests
def f():
    return 42


class C:
    def m(n):
        return 24


class Unpickable(object):
    def __getstate__(self):
        raise pickle.PickleError()


@override_settings(CACHES={
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
    }
})
class DummyCacheTests(TestCase):
    # The Dummy cache backend doesn't really behave like a test backend,
    # so it has its own test case.

    def test_simple(self):
        "Dummy cache backend ignores cache set calls"
        cache.set("key", "value")
        self.assertEqual(cache.get("key"), None)

    def test_add(self):
        "Add doesn't do anything in dummy cache backend"
        cache.add("addkey1", "value")
        result = cache.add("addkey1", "newvalue")
        self.assertEqual(result, True)
        self.assertEqual(cache.get("addkey1"), None)

    def test_non_existent(self):
        "Non-existent keys aren't found in the dummy cache backend"
        self.assertEqual(cache.get("does_not_exist"), None)
        self.assertEqual(cache.get("does_not_exist", "bang!"), "bang!")

    def test_get_many(self):
        "get_many returns nothing for the dummy cache backend"
        cache.set('a', 'a')
        cache.set('b', 'b')
        cache.set('c', 'c')
        cache.set('d', 'd')
        self.assertEqual(cache.get_many(['a', 'c', 'd']), {})
        self.assertEqual(cache.get_many(['a', 'b', 'e']), {})

    def test_delete(self):
        "Cache deletion is transparently ignored on the dummy cache backend"
        cache.set("key1", "spam")
        cache.set("key2", "eggs")
        self.assertEqual(cache.get("key1"), None)
        cache.delete("key1")
        self.assertEqual(cache.get("key1"), None)
        self.assertEqual(cache.get("key2"), None)

    def test_has_key(self):
        "The has_key method doesn't ever return True for the dummy cache backend"
        cache.set("hello1", "goodbye1")
        self.assertEqual(cache.has_key("hello1"), False)
        self.assertEqual(cache.has_key("goodbye1"), False)

    def test_in(self):
        "The in operator doesn't ever return True for the dummy cache backend"
        cache.set("hello2", "goodbye2")
        self.assertEqual("hello2" in cache, False)
        self.assertEqual("goodbye2" in cache, False)

    def test_incr(self):
        "Dummy cache values can't be incremented"
        cache.set('answer', 42)
        self.assertRaises(ValueError, cache.incr, 'answer')
        self.assertRaises(ValueError, cache.incr, 'does_not_exist')

    def test_decr(self):
        "Dummy cache values can't be decremented"
        cache.set('answer', 42)
        self.assertRaises(ValueError, cache.decr, 'answer')
        self.assertRaises(ValueError, cache.decr, 'does_not_exist')

    def test_data_types(self):
        "All data types are ignored equally by the dummy cache"
        stuff = {
            'string': 'this is a string',
            'int': 42,
            'list': [1, 2, 3, 4],
            'tuple': (1, 2, 3, 4),
            'dict': {'A': 1, 'B': 2},
            'function': f,
            'class': C,
        }
        cache.set("stuff", stuff)
        self.assertEqual(cache.get("stuff"), None)

    def test_expiration(self):
        "Expiration has no effect on the dummy cache"
        cache.set('expire1', 'very quickly', 1)
        cache.set('expire2', 'very quickly', 1)
        cache.set('expire3', 'very quickly', 1)

        time.sleep(2)
        self.assertEqual(cache.get("expire1"), None)

        cache.add("expire2", "newvalue")
        self.assertEqual(cache.get("expire2"), None)
        self.assertEqual(cache.has_key("expire3"), False)

    def test_unicode(self):
        "Unicode values are ignored by the dummy cache"
        stuff = {
            'ascii': 'ascii_value',
            'unicode_ascii': 'Iñtërnâtiônàlizætiøn1',
            'Iñtërnâtiônàlizætiøn': 'Iñtërnâtiônàlizætiøn2',
            'ascii2': {'x': 1}
        }
        for (key, value) in stuff.items():
            cache.set(key, value)
            self.assertEqual(cache.get(key), None)

    def test_set_many(self):
        "set_many does nothing for the dummy cache backend"
        cache.set_many({'a': 1, 'b': 2})
        cache.set_many({'a': 1, 'b': 2}, timeout=2, version='1')

    def test_delete_many(self):
        "delete_many does nothing for the dummy cache backend"
        cache.delete_many(['a', 'b'])

    def test_clear(self):
        "clear does nothing for the dummy cache backend"
        cache.clear()

    def test_incr_version(self):
        "Dummy cache versions can't be incremented"
        cache.set('answer', 42)
        self.assertRaises(ValueError, cache.incr_version, 'answer')
        self.assertRaises(ValueError, cache.incr_version, 'does_not_exist')

    def test_decr_version(self):
        "Dummy cache versions can't be decremented"
        cache.set('answer', 42)
        self.assertRaises(ValueError, cache.decr_version, 'answer')
        self.assertRaises(ValueError, cache.decr_version, 'does_not_exist')


def custom_key_func(key, key_prefix, version):
    "A customized cache key function"
    return 'CUSTOM-' + '-'.join([key_prefix, str(version), key])


_caches_setting_base = {
    'default': {},
    'prefix': {'KEY_PREFIX': 'cacheprefix'},
    'v2': {'VERSION': 2},
    'custom_key': {'KEY_FUNCTION': custom_key_func},
    'custom_key2': {'KEY_FUNCTION': 'cache.tests.custom_key_func'},
    'cull': {'OPTIONS': {'MAX_ENTRIES': 30}},
    'zero_cull': {'OPTIONS': {'CULL_FREQUENCY': 0, 'MAX_ENTRIES': 30}},
    'other': {'LOCATION': 'other'},
}


def caches_setting_for_tests(**params):
    setting = dict((k, v.copy()) for k, v in _caches_setting_base.items())
    for cache_params in setting.values():
        cache_params.update(params)
    return setting


class BaseCacheTests(object):
    # A common set of tests to apply to all cache backends

    def setUp(self):
        self.factory = RequestFactory()

    def tearDown(self):
        cache.clear()

    def test_simple(self):
        # Simple cache set/get works
        cache.set("key", "value")
        self.assertEqual(cache.get("key"), "value")

    def test_add(self):
        # A key can be added to a cache
        cache.add("addkey1", "value")
        result = cache.add("addkey1", "newvalue")
        self.assertEqual(result, False)
        self.assertEqual(cache.get("addkey1"), "value")

    def test_prefix(self):
        # Test for same cache key conflicts between shared backend
        cache.set('somekey', 'value')

        # should not be set in the prefixed cache
        self.assertFalse(caches['prefix'].has_key('somekey'))

        caches['prefix'].set('somekey', 'value2')

        self.assertEqual(cache.get('somekey'), 'value')
        self.assertEqual(caches['prefix'].get('somekey'), 'value2')

    def test_non_existent(self):
        # Non-existent cache keys return as None/default
        # get with non-existent keys
        self.assertEqual(cache.get("does_not_exist"), None)
        self.assertEqual(cache.get("does_not_exist", "bang!"), "bang!")

    def test_get_many(self):
        # Multiple cache keys can be returned using get_many
        cache.set('a', 'a')
        cache.set('b', 'b')
        cache.set('c', 'c')
        cache.set('d', 'd')
        self.assertEqual(cache.get_many(['a', 'c', 'd']), {'a': 'a', 'c': 'c', 'd': 'd'})
        self.assertEqual(cache.get_many(['a', 'b', 'e']), {'a': 'a', 'b': 'b'})

    def test_delete(self):
        # Cache keys can be deleted
        cache.set("key1", "spam")
        cache.set("key2", "eggs")
        self.assertEqual(cache.get("key1"), "spam")
        cache.delete("key1")
        self.assertEqual(cache.get("key1"), None)
        self.assertEqual(cache.get("key2"), "eggs")

    def test_has_key(self):
        # The cache can be inspected for cache keys
        cache.set("hello1", "goodbye1")
        self.assertEqual(cache.has_key("hello1"), True)
        self.assertEqual(cache.has_key("goodbye1"), False)

    def test_in(self):
        # The in operator can be used to inspect cache contents
        cache.set("hello2", "goodbye2")
        self.assertEqual("hello2" in cache, True)
        self.assertEqual("goodbye2" in cache, False)

    def test_incr(self):
        # Cache values can be incremented
        cache.set('answer', 41)
        self.assertEqual(cache.incr('answer'), 42)
        self.assertEqual(cache.get('answer'), 42)
        self.assertEqual(cache.incr('answer', 10), 52)
        self.assertEqual(cache.get('answer'), 52)
        self.assertEqual(cache.incr('answer', -10), 42)
        self.assertRaises(ValueError, cache.incr, 'does_not_exist')

    def test_decr(self):
        # Cache values can be decremented
        cache.set('answer', 43)
        self.assertEqual(cache.decr('answer'), 42)
        self.assertEqual(cache.get('answer'), 42)
        self.assertEqual(cache.decr('answer', 10), 32)
        self.assertEqual(cache.get('answer'), 32)
        self.assertEqual(cache.decr('answer', -10), 42)
        self.assertRaises(ValueError, cache.decr, 'does_not_exist')

    def test_close(self):
        self.assertTrue(hasattr(cache, 'close'))
        cache.close()

    def test_data_types(self):
        # Many different data types can be cached
        stuff = {
            'string': 'this is a string',
            'int': 42,
            'list': [1, 2, 3, 4],
            'tuple': (1, 2, 3, 4),
            'dict': {'A': 1, 'B': 2},
            'function': f,
            'class': C,
        }
        cache.set("stuff", stuff)
        self.assertEqual(cache.get("stuff"), stuff)

    def test_cache_read_for_model_instance(self):
        # Don't want fields with callable as default to be called on cache read
        expensive_calculation.num_runs = 0
        Poll.objects.all().delete()
        my_poll = Poll.objects.create(question="Well?")
        self.assertEqual(Poll.objects.count(), 1)
        pub_date = my_poll.pub_date
        cache.set('question', my_poll)
        cached_poll = cache.get('question')
        self.assertEqual(cached_poll.pub_date, pub_date)
        # We only want the default expensive calculation run once
        self.assertEqual(expensive_calculation.num_runs, 1)

    def test_cache_write_for_model_instance_with_deferred(self):
        # Don't want fields with callable as default to be called on cache write
        expensive_calculation.num_runs = 0
        Poll.objects.all().delete()
        Poll.objects.create(question="What?")
        self.assertEqual(expensive_calculation.num_runs, 1)
        defer_qs = Poll.objects.all().defer('question')
        self.assertEqual(defer_qs.count(), 1)
        self.assertEqual(expensive_calculation.num_runs, 1)
        cache.set('deferred_queryset', defer_qs)
        # cache set should not re-evaluate default functions
        self.assertEqual(expensive_calculation.num_runs, 1)

    def test_cache_read_for_model_instance_with_deferred(self):
        # Don't want fields with callable as default to be called on cache read
        expensive_calculation.num_runs = 0
        Poll.objects.all().delete()
        Poll.objects.create(question="What?")
        self.assertEqual(expensive_calculation.num_runs, 1)
        defer_qs = Poll.objects.all().defer('question')
        self.assertEqual(defer_qs.count(), 1)
        cache.set('deferred_queryset', defer_qs)
        self.assertEqual(expensive_calculation.num_runs, 1)
        runs_before_cache_read = expensive_calculation.num_runs
        cache.get('deferred_queryset')
        # We only want the default expensive calculation run on creation and set
        self.assertEqual(expensive_calculation.num_runs, runs_before_cache_read)

    def test_expiration(self):
        # Cache values can be set to expire
        cache.set('expire1', 'very quickly', 1)
        cache.set('expire2', 'very quickly', 1)
        cache.set('expire3', 'very quickly', 1)

        time.sleep(2)
        self.assertEqual(cache.get("expire1"), None)

        cache.add("expire2", "newvalue")
        self.assertEqual(cache.get("expire2"), "newvalue")
        self.assertEqual(cache.has_key("expire3"), False)

    def test_unicode(self):
        # Unicode values can be cached
        stuff = {
            'ascii': 'ascii_value',
            'unicode_ascii': 'Iñtërnâtiônàlizætiøn1',
            'Iñtërnâtiônàlizætiøn': 'Iñtërnâtiônàlizætiøn2',
            'ascii2': {'x': 1}
        }
        # Test `set`
        for (key, value) in stuff.items():
            cache.set(key, value)
            self.assertEqual(cache.get(key), value)

        # Test `add`
        for (key, value) in stuff.items():
            cache.delete(key)
            cache.add(key, value)
            self.assertEqual(cache.get(key), value)

        # Test `set_many`
        for (key, value) in stuff.items():
            cache.delete(key)
        cache.set_many(stuff)
        for (key, value) in stuff.items():
            self.assertEqual(cache.get(key), value)

    def test_binary_string(self):
        # Binary strings should be cacheable
        from zlib import compress, decompress
        value = 'value_to_be_compressed'
        compressed_value = compress(value.encode())

        # Test set
        cache.set('binary1', compressed_value)
        compressed_result = cache.get('binary1')
        self.assertEqual(compressed_value, compressed_result)
        self.assertEqual(value, decompress(compressed_result).decode())

        # Test add
        cache.add('binary1-add', compressed_value)
        compressed_result = cache.get('binary1-add')
        self.assertEqual(compressed_value, compressed_result)
        self.assertEqual(value, decompress(compressed_result).decode())

        # Test set_many
        cache.set_many({'binary1-set_many': compressed_value})
        compressed_result = cache.get('binary1-set_many')
        self.assertEqual(compressed_value, compressed_result)
        self.assertEqual(value, decompress(compressed_result).decode())

    def test_set_many(self):
        # Multiple keys can be set using set_many
        cache.set_many({"key1": "spam", "key2": "eggs"})
        self.assertEqual(cache.get("key1"), "spam")
        self.assertEqual(cache.get("key2"), "eggs")

    def test_set_many_expiration(self):
        # set_many takes a second ``timeout`` parameter
        cache.set_many({"key1": "spam", "key2": "eggs"}, 1)
        time.sleep(2)
        self.assertEqual(cache.get("key1"), None)
        self.assertEqual(cache.get("key2"), None)

    def test_delete_many(self):
        # Multiple keys can be deleted using delete_many
        cache.set("key1", "spam")
        cache.set("key2", "eggs")
        cache.set("key3", "ham")
        cache.delete_many(["key1", "key2"])
        self.assertEqual(cache.get("key1"), None)
        self.assertEqual(cache.get("key2"), None)
        self.assertEqual(cache.get("key3"), "ham")

    def test_clear(self):
        # The cache can be emptied using clear
        cache.set("key1", "spam")
        cache.set("key2", "eggs")
        cache.clear()
        self.assertEqual(cache.get("key1"), None)
        self.assertEqual(cache.get("key2"), None)

    def test_long_timeout(self):
        '''
        Using a timeout greater than 30 days makes memcached think
        it is an absolute expiration timestamp instead of a relative
        offset. Test that we honour this convention. Refs #12399.
        '''
        cache.set('key1', 'eggs', 60 * 60 * 24 * 30 + 1)  # 30 days + 1 second
        self.assertEqual(cache.get('key1'), 'eggs')

        cache.add('key2', 'ham', 60 * 60 * 24 * 30 + 1)
        self.assertEqual(cache.get('key2'), 'ham')

        cache.set_many({'key3': 'sausage', 'key4': 'lobster bisque'}, 60 * 60 * 24 * 30 + 1)
        self.assertEqual(cache.get('key3'), 'sausage')
        self.assertEqual(cache.get('key4'), 'lobster bisque')

    def test_forever_timeout(self):
        '''
        Passing in None into timeout results in a value that is cached forever
        '''
        cache.set('key1', 'eggs', None)
        self.assertEqual(cache.get('key1'), 'eggs')

        cache.add('key2', 'ham', None)
        self.assertEqual(cache.get('key2'), 'ham')

        cache.set_many({'key3': 'sausage', 'key4': 'lobster bisque'}, None)
        self.assertEqual(cache.get('key3'), 'sausage')
        self.assertEqual(cache.get('key4'), 'lobster bisque')

    def test_zero_timeout(self):
        '''
        Passing in None into timeout results in a value that is cached forever
        '''
        cache.set('key1', 'eggs', 0)
        self.assertEqual(cache.get('key1'), None)

        cache.add('key2', 'ham', 0)
        self.assertEqual(cache.get('key2'), None)

        cache.set_many({'key3': 'sausage', 'key4': 'lobster bisque'}, 0)
        self.assertEqual(cache.get('key3'), None)
        self.assertEqual(cache.get('key4'), None)

    def test_float_timeout(self):
        # Make sure a timeout given as a float doesn't crash anything.
        cache.set("key1", "spam", 100.2)
        self.assertEqual(cache.get("key1"), "spam")

    def _perform_cull_test(self, cull_cache, initial_count, final_count):
        # Create initial cache key entries. This will overflow the cache,
        # causing a cull.
        for i in range(1, initial_count):
            cull_cache.set('cull%d' % i, 'value', 1000)
        count = 0
        # Count how many keys are left in the cache.
        for i in range(1, initial_count):
            if cull_cache.has_key('cull%d' % i):
                count = count + 1
        self.assertEqual(count, final_count)

    def test_cull(self):
        self._perform_cull_test(caches['cull'], 50, 29)

    def test_zero_cull(self):
        self._perform_cull_test(caches['zero_cull'], 50, 19)

    def test_invalid_keys(self):
        """
        All the builtin backends (except memcached, see below) should warn on
        keys that would be refused by memcached. This encourages portable
        caching code without making it too difficult to use production backends
        with more liberal key rules. Refs #6447.

        """
        # mimic custom ``make_key`` method being defined since the default will
        # never show the below warnings
        def func(key, *args):
            return key

        old_func = cache.key_func
        cache.key_func = func

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # memcached does not allow whitespace or control characters in keys
                cache.set('key with spaces', 'value')
                self.assertEqual(len(w), 2)
                self.assertIsInstance(w[0].message, CacheKeyWarning)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # memcached limits key length to 250
                cache.set('a' * 251, 'value')
                self.assertEqual(len(w), 1)
                self.assertIsInstance(w[0].message, CacheKeyWarning)
        finally:
            cache.key_func = old_func

    def test_cache_versioning_get_set(self):
        # set, using default version = 1
        cache.set('answer1', 42)
        self.assertEqual(cache.get('answer1'), 42)
        self.assertEqual(cache.get('answer1', version=1), 42)
        self.assertEqual(cache.get('answer1', version=2), None)

        self.assertEqual(caches['v2'].get('answer1'), None)
        self.assertEqual(caches['v2'].get('answer1', version=1), 42)
        self.assertEqual(caches['v2'].get('answer1', version=2), None)

        # set, default version = 1, but manually override version = 2
        cache.set('answer2', 42, version=2)
        self.assertEqual(cache.get('answer2'), None)
        self.assertEqual(cache.get('answer2', version=1), None)
        self.assertEqual(cache.get('answer2', version=2), 42)

        self.assertEqual(caches['v2'].get('answer2'), 42)
        self.assertEqual(caches['v2'].get('answer2', version=1), None)
        self.assertEqual(caches['v2'].get('answer2', version=2), 42)

        # v2 set, using default version = 2
        caches['v2'].set('answer3', 42)
        self.assertEqual(cache.get('answer3'), None)
        self.assertEqual(cache.get('answer3', version=1), None)
        self.assertEqual(cache.get('answer3', version=2), 42)

        self.assertEqual(caches['v2'].get('answer3'), 42)
        self.assertEqual(caches['v2'].get('answer3', version=1), None)
        self.assertEqual(caches['v2'].get('answer3', version=2), 42)

        # v2 set, default version = 2, but manually override version = 1
        caches['v2'].set('answer4', 42, version=1)
        self.assertEqual(cache.get('answer4'), 42)
        self.assertEqual(cache.get('answer4', version=1), 42)
        self.assertEqual(cache.get('answer4', version=2), None)

        self.assertEqual(caches['v2'].get('answer4'), None)
        self.assertEqual(caches['v2'].get('answer4', version=1), 42)
        self.assertEqual(caches['v2'].get('answer4', version=2), None)

    def test_cache_versioning_add(self):

        # add, default version = 1, but manually override version = 2
        cache.add('answer1', 42, version=2)
        self.assertEqual(cache.get('answer1', version=1), None)
        self.assertEqual(cache.get('answer1', version=2), 42)

        cache.add('answer1', 37, version=2)
        self.assertEqual(cache.get('answer1', version=1), None)
        self.assertEqual(cache.get('answer1', version=2), 42)

        cache.add('answer1', 37, version=1)
        self.assertEqual(cache.get('answer1', version=1), 37)
        self.assertEqual(cache.get('answer1', version=2), 42)

        # v2 add, using default version = 2
        caches['v2'].add('answer2', 42)
        self.assertEqual(cache.get('answer2', version=1), None)
        self.assertEqual(cache.get('answer2', version=2), 42)

        caches['v2'].add('answer2', 37)
        self.assertEqual(cache.get('answer2', version=1), None)
        self.assertEqual(cache.get('answer2', version=2), 42)

        caches['v2'].add('answer2', 37, version=1)
        self.assertEqual(cache.get('answer2', version=1), 37)
        self.assertEqual(cache.get('answer2', version=2), 42)

        # v2 add, default version = 2, but manually override version = 1
        caches['v2'].add('answer3', 42, version=1)
        self.assertEqual(cache.get('answer3', version=1), 42)
        self.assertEqual(cache.get('answer3', version=2), None)

        caches['v2'].add('answer3', 37, version=1)
        self.assertEqual(cache.get('answer3', version=1), 42)
        self.assertEqual(cache.get('answer3', version=2), None)

        caches['v2'].add('answer3', 37)
        self.assertEqual(cache.get('answer3', version=1), 42)
        self.assertEqual(cache.get('answer3', version=2), 37)

    def test_cache_versioning_has_key(self):
        cache.set('answer1', 42)

        # has_key
        self.assertTrue(cache.has_key('answer1'))
        self.assertTrue(cache.has_key('answer1', version=1))
        self.assertFalse(cache.has_key('answer1', version=2))

        self.assertFalse(caches['v2'].has_key('answer1'))
        self.assertTrue(caches['v2'].has_key('answer1', version=1))
        self.assertFalse(caches['v2'].has_key('answer1', version=2))

    def test_cache_versioning_delete(self):
        cache.set('answer1', 37, version=1)
        cache.set('answer1', 42, version=2)
        cache.delete('answer1')
        self.assertEqual(cache.get('answer1', version=1), None)
        self.assertEqual(cache.get('answer1', version=2), 42)

        cache.set('answer2', 37, version=1)
        cache.set('answer2', 42, version=2)
        cache.delete('answer2', version=2)
        self.assertEqual(cache.get('answer2', version=1), 37)
        self.assertEqual(cache.get('answer2', version=2), None)

        cache.set('answer3', 37, version=1)
        cache.set('answer3', 42, version=2)
        caches['v2'].delete('answer3')
        self.assertEqual(cache.get('answer3', version=1), 37)
        self.assertEqual(cache.get('answer3', version=2), None)

        cache.set('answer4', 37, version=1)
        cache.set('answer4', 42, version=2)
        caches['v2'].delete('answer4', version=1)
        self.assertEqual(cache.get('answer4', version=1), None)
        self.assertEqual(cache.get('answer4', version=2), 42)

    def test_cache_versioning_incr_decr(self):
        cache.set('answer1', 37, version=1)
        cache.set('answer1', 42, version=2)
        cache.incr('answer1')
        self.assertEqual(cache.get('answer1', version=1), 38)
        self.assertEqual(cache.get('answer1', version=2), 42)
        cache.decr('answer1')
        self.assertEqual(cache.get('answer1', version=1), 37)
        self.assertEqual(cache.get('answer1', version=2), 42)

        cache.set('answer2', 37, version=1)
        cache.set('answer2', 42, version=2)
        cache.incr('answer2', version=2)
        self.assertEqual(cache.get('answer2', version=1), 37)
        self.assertEqual(cache.get('answer2', version=2), 43)
        cache.decr('answer2', version=2)
        self.assertEqual(cache.get('answer2', version=1), 37)
        self.assertEqual(cache.get('answer2', version=2), 42)

        cache.set('answer3', 37, version=1)
        cache.set('answer3', 42, version=2)
        caches['v2'].incr('answer3')
        self.assertEqual(cache.get('answer3', version=1), 37)
        self.assertEqual(cache.get('answer3', version=2), 43)
        caches['v2'].decr('answer3')
        self.assertEqual(cache.get('answer3', version=1), 37)
        self.assertEqual(cache.get('answer3', version=2), 42)

        cache.set('answer4', 37, version=1)
        cache.set('answer4', 42, version=2)
        caches['v2'].incr('answer4', version=1)
        self.assertEqual(cache.get('answer4', version=1), 38)
        self.assertEqual(cache.get('answer4', version=2), 42)
        caches['v2'].decr('answer4', version=1)
        self.assertEqual(cache.get('answer4', version=1), 37)
        self.assertEqual(cache.get('answer4', version=2), 42)

    def test_cache_versioning_get_set_many(self):
        # set, using default version = 1
        cache.set_many({'ford1': 37, 'arthur1': 42})
        self.assertEqual(cache.get_many(['ford1', 'arthur1']),
                         {'ford1': 37, 'arthur1': 42})
        self.assertEqual(cache.get_many(['ford1', 'arthur1'], version=1),
                         {'ford1': 37, 'arthur1': 42})
        self.assertEqual(cache.get_many(['ford1', 'arthur1'], version=2), {})

        self.assertEqual(caches['v2'].get_many(['ford1', 'arthur1']), {})
        self.assertEqual(caches['v2'].get_many(['ford1', 'arthur1'], version=1),
                         {'ford1': 37, 'arthur1': 42})
        self.assertEqual(caches['v2'].get_many(['ford1', 'arthur1'], version=2), {})

        # set, default version = 1, but manually override version = 2
        cache.set_many({'ford2': 37, 'arthur2': 42}, version=2)
        self.assertEqual(cache.get_many(['ford2', 'arthur2']), {})
        self.assertEqual(cache.get_many(['ford2', 'arthur2'], version=1), {})
        self.assertEqual(cache.get_many(['ford2', 'arthur2'], version=2),
                         {'ford2': 37, 'arthur2': 42})

        self.assertEqual(caches['v2'].get_many(['ford2', 'arthur2']),
                         {'ford2': 37, 'arthur2': 42})
        self.assertEqual(caches['v2'].get_many(['ford2', 'arthur2'], version=1), {})
        self.assertEqual(caches['v2'].get_many(['ford2', 'arthur2'], version=2),
                         {'ford2': 37, 'arthur2': 42})

        # v2 set, using default version = 2
        caches['v2'].set_many({'ford3': 37, 'arthur3': 42})
        self.assertEqual(cache.get_many(['ford3', 'arthur3']), {})
        self.assertEqual(cache.get_many(['ford3', 'arthur3'], version=1), {})
        self.assertEqual(cache.get_many(['ford3', 'arthur3'], version=2),
                         {'ford3': 37, 'arthur3': 42})

        self.assertEqual(caches['v2'].get_many(['ford3', 'arthur3']),
                         {'ford3': 37, 'arthur3': 42})
        self.assertEqual(caches['v2'].get_many(['ford3', 'arthur3'], version=1), {})
        self.assertEqual(caches['v2'].get_many(['ford3', 'arthur3'], version=2),
                         {'ford3': 37, 'arthur3': 42})

        # v2 set, default version = 2, but manually override version = 1
        caches['v2'].set_many({'ford4': 37, 'arthur4': 42}, version=1)
        self.assertEqual(cache.get_many(['ford4', 'arthur4']),
                         {'ford4': 37, 'arthur4': 42})
        self.assertEqual(cache.get_many(['ford4', 'arthur4'], version=1),
                         {'ford4': 37, 'arthur4': 42})
        self.assertEqual(cache.get_many(['ford4', 'arthur4'], version=2), {})

        self.assertEqual(caches['v2'].get_many(['ford4', 'arthur4']), {})
        self.assertEqual(caches['v2'].get_many(['ford4', 'arthur4'], version=1),
                         {'ford4': 37, 'arthur4': 42})
        self.assertEqual(caches['v2'].get_many(['ford4', 'arthur4'], version=2), {})

    def test_incr_version(self):
        cache.set('answer', 42, version=2)
        self.assertEqual(cache.get('answer'), None)
        self.assertEqual(cache.get('answer', version=1), None)
        self.assertEqual(cache.get('answer', version=2), 42)
        self.assertEqual(cache.get('answer', version=3), None)

        self.assertEqual(cache.incr_version('answer', version=2), 3)
        self.assertEqual(cache.get('answer'), None)
        self.assertEqual(cache.get('answer', version=1), None)
        self.assertEqual(cache.get('answer', version=2), None)
        self.assertEqual(cache.get('answer', version=3), 42)

        caches['v2'].set('answer2', 42)
        self.assertEqual(caches['v2'].get('answer2'), 42)
        self.assertEqual(caches['v2'].get('answer2', version=1), None)
        self.assertEqual(caches['v2'].get('answer2', version=2), 42)
        self.assertEqual(caches['v2'].get('answer2', version=3), None)

        self.assertEqual(caches['v2'].incr_version('answer2'), 3)
        self.assertEqual(caches['v2'].get('answer2'), None)
        self.assertEqual(caches['v2'].get('answer2', version=1), None)
        self.assertEqual(caches['v2'].get('answer2', version=2), None)
        self.assertEqual(caches['v2'].get('answer2', version=3), 42)

        self.assertRaises(ValueError, cache.incr_version, 'does_not_exist')

    def test_decr_version(self):
        cache.set('answer', 42, version=2)
        self.assertEqual(cache.get('answer'), None)
        self.assertEqual(cache.get('answer', version=1), None)
        self.assertEqual(cache.get('answer', version=2), 42)

        self.assertEqual(cache.decr_version('answer', version=2), 1)
        self.assertEqual(cache.get('answer'), 42)
        self.assertEqual(cache.get('answer', version=1), 42)
        self.assertEqual(cache.get('answer', version=2), None)

        caches['v2'].set('answer2', 42)
        self.assertEqual(caches['v2'].get('answer2'), 42)
        self.assertEqual(caches['v2'].get('answer2', version=1), None)
        self.assertEqual(caches['v2'].get('answer2', version=2), 42)

        self.assertEqual(caches['v2'].decr_version('answer2'), 1)
        self.assertEqual(caches['v2'].get('answer2'), None)
        self.assertEqual(caches['v2'].get('answer2', version=1), 42)
        self.assertEqual(caches['v2'].get('answer2', version=2), None)

        self.assertRaises(ValueError, cache.decr_version, 'does_not_exist', version=2)

    def test_custom_key_func(self):
        # Two caches with different key functions aren't visible to each other
        cache.set('answer1', 42)
        self.assertEqual(cache.get('answer1'), 42)
        self.assertEqual(caches['custom_key'].get('answer1'), None)
        self.assertEqual(caches['custom_key2'].get('answer1'), None)

        caches['custom_key'].set('answer2', 42)
        self.assertEqual(cache.get('answer2'), None)
        self.assertEqual(caches['custom_key'].get('answer2'), 42)
        self.assertEqual(caches['custom_key2'].get('answer2'), 42)

    def test_cache_write_unpickable_object(self):
        update_middleware = UpdateCacheMiddleware()
        update_middleware.cache = cache

        fetch_middleware = FetchFromCacheMiddleware()
        fetch_middleware.cache = cache

        request = self.factory.get('/cache/test')
        request._cache_update_cache = True
        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        self.assertEqual(get_cache_data, None)

        response = HttpResponse()
        content = 'Testing cookie serialization.'
        response.content = content
        response.set_cookie('foo', 'bar')

        update_middleware.process_response(request, response)

        get_cache_data = fetch_middleware.process_request(request)
        self.assertNotEqual(get_cache_data, None)
        self.assertEqual(get_cache_data.content, content.encode('utf-8'))
        self.assertEqual(get_cache_data.cookies, response.cookies)

        update_middleware.process_response(request, get_cache_data)
        get_cache_data = fetch_middleware.process_request(request)
        self.assertNotEqual(get_cache_data, None)
        self.assertEqual(get_cache_data.content, content.encode('utf-8'))
        self.assertEqual(get_cache_data.cookies, response.cookies)

    def test_add_fail_on_pickleerror(self):
         "See https://code.djangoproject.com/ticket/21200"
         with self.assertRaises(pickle.PickleError):
             cache.add('unpickable', Unpickable())

    def test_set_fail_on_pickleerror(self):
        "See https://code.djangoproject.com/ticket/21200"
        with self.assertRaises(pickle.PickleError):
            cache.set('unpickable', Unpickable())


@override_settings(CACHES=caches_setting_for_tests(
    BACKEND='django.core.cache.backends.db.DatabaseCache',
    # Spaces are used in the table name to ensure quoting/escaping is working
    LOCATION='test cache table'
))
class DBCacheTests(BaseCacheTests, TransactionTestCase):

    available_apps = ['cache']

    def setUp(self):
        # The super calls needs to happen first for the settings override.
        super(DBCacheTests, self).setUp()
        self.create_table()

    def tearDown(self):
        # The super call needs to happen first because it uses the database.
        super(DBCacheTests, self).tearDown()
        self.drop_table()

    def create_table(self):
        management.call_command('createcachetable', verbosity=0, interactive=False)

    def drop_table(self):
        cursor = connection.cursor()
        table_name = connection.ops.quote_name('test cache table')
        cursor.execute('DROP TABLE %s' % table_name)
        cursor.close()

    def test_zero_cull(self):
        self._perform_cull_test(caches['zero_cull'], 50, 18)

    def test_second_call_doesnt_crash(self):
        stdout = six.StringIO()
        management.call_command(
            'createcachetable',
            stdout=stdout
        )
        self.assertEqual(stdout.getvalue(),
            "Cache table 'test cache table' already exists.\n" * len(settings.CACHES))

    def test_createcachetable_with_table_argument(self):
        """
        Delete and recreate cache table with legacy behavior (explicitly
        specifying the table name).
        """
        self.drop_table()
        stdout = six.StringIO()
        management.call_command(
            'createcachetable',
            'test cache table',
            verbosity=2,
            stdout=stdout
        )
        self.assertEqual(stdout.getvalue(),
            "Cache table 'test cache table' created.\n")

    def test_clear_commits_transaction(self):
        # Ensure the database transaction is committed (#19896)
        cache.set("key1", "spam")
        cache.clear()
        transaction.rollback()
        self.assertEqual(cache.get("key1"), None)


@override_settings(USE_TZ=True)
class DBCacheWithTimeZoneTests(DBCacheTests):
    pass


class DBCacheRouter(object):
    """A router that puts the cache table on the 'other' database."""

    def db_for_read(self, model, **hints):
        if model._meta.app_label == 'django_cache':
            return 'other'

    def db_for_write(self, model, **hints):
        if model._meta.app_label == 'django_cache':
            return 'other'

    def allow_migrate(self, db, model):
        if model._meta.app_label == 'django_cache':
            return db == 'other'


@override_settings(
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
            'LOCATION': 'my_cache_table',
        },
    },
)
class CreateCacheTableForDBCacheTests(TestCase):
    multi_db = True

    def test_createcachetable_observes_database_router(self):
        old_routers = router.routers
        try:
            router.routers = [DBCacheRouter()]
            # cache table should not be created on 'default'
            with self.assertNumQueries(0, using='default'):
                management.call_command('createcachetable',
                                        database='default',
                                        verbosity=0, interactive=False)
            # cache table should be created on 'other'
            # Queries:
            #   1: check table doesn't already exist
            #   2: create the table
            #   3: create the index
            with self.assertNumQueries(3, using='other'):
                management.call_command('createcachetable',
                                        database='other',
                                        verbosity=0, interactive=False)
        finally:
            router.routers = old_routers


@override_settings(CACHES=caches_setting_for_tests(
    BACKEND='django.core.cache.backends.locmem.LocMemCache',
))
class LocMemCacheTests(BaseCacheTests, TestCase):

    def setUp(self):
        super(LocMemCacheTests, self).setUp()

        # LocMem requires a hack to make the other caches
        # share a data store with the 'normal' cache.
        caches['prefix']._cache = cache._cache
        caches['prefix']._expire_info = cache._expire_info

        caches['v2']._cache = cache._cache
        caches['v2']._expire_info = cache._expire_info

        caches['custom_key']._cache = cache._cache
        caches['custom_key']._expire_info = cache._expire_info

        caches['custom_key2']._cache = cache._cache
        caches['custom_key2']._expire_info = cache._expire_info

    def test_multiple_caches(self):
        "Check that multiple locmem caches are isolated"
        cache.set('value', 42)
        self.assertEqual(caches['default'].get('value'), 42)
        self.assertEqual(caches['other'].get('value'), None)

    def test_incr_decr_timeout(self):
        """incr/decr does not modify expiry time (matches memcached behavior)"""
        key = 'value'
        _key = cache.make_key(key)
        cache.set(key, 1, timeout=cache.default_timeout * 10)
        expire = cache._expire_info[_key]
        cache.incr(key)
        self.assertEqual(expire, cache._expire_info[_key])
        cache.decr(key)
        self.assertEqual(expire, cache._expire_info[_key])


# memcached backend isn't guaranteed to be available.
# To check the memcached backend, the test settings file will
# need to contain at least one cache backend setting that points at
# your memcache server.
memcached_params = {}
for _cache_params in settings.CACHES.values():
    if _cache_params['BACKEND'].startswith('django.core.cache.backends.memcached.'):
        memcached_params = _cache_params


@unittest.skipUnless(memcached_params, "memcached not available")
@override_settings(CACHES=caches_setting_for_tests(**memcached_params))
class MemcachedCacheTests(BaseCacheTests, TestCase):

    def test_invalid_keys(self):
        """
        On memcached, we don't introduce a duplicate key validation
        step (for speed reasons), we just let the memcached API
        library raise its own exception on bad keys. Refs #6447.

        In order to be memcached-API-library agnostic, we only assert
        that a generic exception of some kind is raised.

        """
        # memcached does not allow whitespace or control characters in keys
        self.assertRaises(Exception, cache.set, 'key with spaces', 'value')
        # memcached limits key length to 250
        self.assertRaises(Exception, cache.set, 'a' * 251, 'value')

    # Explicitly display a skipped test if no configured cache uses MemcachedCache
    @unittest.skipUnless(
        any(c['BACKEND'] == 'django.core.cache.backends.memcached.MemcachedCache'
            for c in settings.CACHES.values()),
        "cache with python-memcached library not available")
    def test_memcached_uses_highest_pickle_version(self):
        # Regression test for #19810
        for cache_key, cache in settings.CACHES.items():
            if cache['BACKEND'] == 'django.core.cache.backends.memcached.MemcachedCache':
                self.assertEqual(caches[cache_key]._cache.pickleProtocol,
                                 pickle.HIGHEST_PROTOCOL)

    def test_cull(self):
        # culling isn't implemented, memcached deals with it.
        pass

    def test_zero_cull(self):
        # culling isn't implemented, memcached deals with it.
        pass


@override_settings(CACHES=caches_setting_for_tests(
    BACKEND='django.core.cache.backends.filebased.FileBasedCache',
))
class FileBasedCacheTests(BaseCacheTests, TestCase):
    """
    Specific test cases for the file-based cache.
    """

    def setUp(self):
        super(FileBasedCacheTests, self).setUp()
        self.dirname = tempfile.mkdtemp()
        for cache_params in settings.CACHES.values():
            cache_params.update({'LOCATION': self.dirname})

    def tearDown(self):
        shutil.rmtree(self.dirname)
        super(FileBasedCacheTests, self).tearDown()

    def test_ignores_non_cache_files(self):
        fname = os.path.join(self.dirname, 'not-a-cache-file')
        with open(fname, 'w'):
            os.utime(fname, None)
        cache.clear()
        self.assertTrue(os.path.exists(fname),
                        'Expected cache.clear to ignore non cache files')
        os.remove(fname)

    def test_clear_does_not_remove_cache_dir(self):
        cache.clear()
        self.assertTrue(os.path.exists(self.dirname),
                        'Expected cache.clear to keep the cache dir')

    def test_creates_cache_dir_if_nonexistent(self):
        os.rmdir(self.dirname)
        cache.set('foo', 'bar')
        os.path.exists(self.dirname)


@override_settings(CACHES={
    'default': {
        'BACKEND': 'cache.liberal_backend.CacheClass',
    },
})
class CustomCacheKeyValidationTests(TestCase):
    """
    Tests for the ability to mixin a custom ``validate_key`` method to
    a custom cache backend that otherwise inherits from a builtin
    backend, and override the default key validation. Refs #6447.

    """
    def test_custom_key_validation(self):
        # this key is both longer than 250 characters, and has spaces
        key = 'some key with spaces' * 15
        val = 'a value'
        cache.set(key, val)
        self.assertEqual(cache.get(key), val)


class GetCacheTests(IgnorePendingDeprecationWarningsMixin, TestCase):

    def test_simple(self):
        from django.core.cache import caches, DEFAULT_CACHE_ALIAS, get_cache
        self.assertIsInstance(
            caches[DEFAULT_CACHE_ALIAS],
            get_cache('default').__class__
        )

        cache = get_cache(
            'django.core.cache.backends.dummy.DummyCache',
            **{'TIMEOUT': 120}
        )
        self.assertEqual(cache.default_timeout, 120)

        self.assertRaises(InvalidCacheBackendError, get_cache, 'does_not_exist')

    def test_close(self):
        from django.core.cache import get_cache
        from django.core import signals
        cache = get_cache('cache.closeable_cache.CacheClass')
        self.assertFalse(cache.closed)
        signals.request_finished.send(self.__class__)
        self.assertTrue(cache.closed)


@override_settings(
    CACHE_MIDDLEWARE_KEY_PREFIX='settingsprefix',
    CACHE_MIDDLEWARE_SECONDS=1,
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        },
    },
    USE_I18N=False,
)
class CacheUtils(TestCase):
    """TestCase for django.utils.cache functions."""

    def setUp(self):
        self.path = '/cache/test/'
        self.factory = RequestFactory()

    def test_patch_vary_headers(self):
        headers = (
            # Initial vary, new headers, resulting vary.
            (None, ('Accept-Encoding',), 'Accept-Encoding'),
            ('Accept-Encoding', ('accept-encoding',), 'Accept-Encoding'),
            ('Accept-Encoding', ('ACCEPT-ENCODING',), 'Accept-Encoding'),
            ('Cookie', ('Accept-Encoding',), 'Cookie, Accept-Encoding'),
            ('Cookie, Accept-Encoding', ('Accept-Encoding',), 'Cookie, Accept-Encoding'),
            ('Cookie, Accept-Encoding', ('Accept-Encoding', 'cookie'), 'Cookie, Accept-Encoding'),
            (None, ('Accept-Encoding', 'COOKIE'), 'Accept-Encoding, COOKIE'),
            ('Cookie,     Accept-Encoding', ('Accept-Encoding', 'cookie'), 'Cookie, Accept-Encoding'),
            ('Cookie    ,     Accept-Encoding', ('Accept-Encoding', 'cookie'), 'Cookie, Accept-Encoding'),
        )
        for initial_vary, newheaders, resulting_vary in headers:
            response = HttpResponse()
            if initial_vary is not None:
                response['Vary'] = initial_vary
            patch_vary_headers(response, newheaders)
            self.assertEqual(response['Vary'], resulting_vary)

    def test_get_cache_key(self):
        request = self.factory.get(self.path)
        response = HttpResponse()
        key_prefix = 'localprefix'
        # Expect None if no headers have been set yet.
        self.assertEqual(get_cache_key(request), None)
        # Set headers to an empty list.
        learn_cache_key(request, response)
        self.assertEqual(get_cache_key(request), 'views.decorators.cache.cache_page.settingsprefix.GET.9fa0fd092afb73bdce204bb4f94d5804.d41d8cd98f00b204e9800998ecf8427e')
        # Verify that a specified key_prefix is taken into account.
        learn_cache_key(request, response, key_prefix=key_prefix)
        self.assertEqual(get_cache_key(request, key_prefix=key_prefix), 'views.decorators.cache.cache_page.localprefix.GET.9fa0fd092afb73bdce204bb4f94d5804.d41d8cd98f00b204e9800998ecf8427e')

    def test_get_cache_key_with_query(self):
        request = self.factory.get(self.path, {'test': 1})
        response = HttpResponse()
        # Expect None if no headers have been set yet.
        self.assertEqual(get_cache_key(request), None)
        # Set headers to an empty list.
        learn_cache_key(request, response)
        # Verify that the querystring is taken into account.
        self.assertEqual(get_cache_key(request), 'views.decorators.cache.cache_page.settingsprefix.GET.d11198ba31883732b0de5786a80cc12b.d41d8cd98f00b204e9800998ecf8427e')

    def test_learn_cache_key(self):
        request = self.factory.head(self.path)
        response = HttpResponse()
        response['Vary'] = 'Pony'
        # Make sure that the Vary header is added to the key hash
        learn_cache_key(request, response)
        self.assertEqual(get_cache_key(request), 'views.decorators.cache.cache_page.settingsprefix.GET.9fa0fd092afb73bdce204bb4f94d5804.d41d8cd98f00b204e9800998ecf8427e')

    def test_patch_cache_control(self):
        tests = (
            # Initial Cache-Control, kwargs to patch_cache_control, expected Cache-Control parts
            (None, {'private': True}, set(['private'])),

            # Test whether private/public attributes are mutually exclusive
            ('private', {'private': True}, set(['private'])),
            ('private', {'public': True}, set(['public'])),
            ('public', {'public': True}, set(['public'])),
            ('public', {'private': True}, set(['private'])),
            ('must-revalidate,max-age=60,private', {'public': True}, set(['must-revalidate', 'max-age=60', 'public'])),
            ('must-revalidate,max-age=60,public', {'private': True}, set(['must-revalidate', 'max-age=60', 'private'])),
            ('must-revalidate,max-age=60', {'public': True}, set(['must-revalidate', 'max-age=60', 'public'])),
        )

        cc_delim_re = re.compile(r'\s*,\s*')

        for initial_cc, newheaders, expected_cc in tests:
            response = HttpResponse()
            if initial_cc is not None:
                response['Cache-Control'] = initial_cc
            patch_cache_control(response, **newheaders)
            parts = set(cc_delim_re.split(response['Cache-Control']))
            self.assertEqual(parts, expected_cc)


@override_settings(
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
            'KEY_PREFIX': 'cacheprefix',
        },
    },
)
class PrefixedCacheUtils(CacheUtils):
    pass


@override_settings(
    CACHE_MIDDLEWARE_SECONDS=60,
    CACHE_MIDDLEWARE_KEY_PREFIX='test',
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        },
    },
)
class CacheHEADTest(TestCase):

    def setUp(self):
        self.path = '/cache/test/'
        self.factory = RequestFactory()

    def tearDown(self):
        cache.clear()

    def _set_cache(self, request, msg):
        response = HttpResponse()
        response.content = msg
        return UpdateCacheMiddleware().process_response(request, response)

    def test_head_caches_correctly(self):
        test_content = 'test content'

        request = self.factory.head(self.path)
        request._cache_update_cache = True
        self._set_cache(request, test_content)

        request = self.factory.head(self.path)
        request._cache_update_cache = True
        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        self.assertNotEqual(get_cache_data, None)
        self.assertEqual(test_content.encode(), get_cache_data.content)

    def test_head_with_cached_get(self):
        test_content = 'test content'

        request = self.factory.get(self.path)
        request._cache_update_cache = True
        self._set_cache(request, test_content)

        request = self.factory.head(self.path)
        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        self.assertNotEqual(get_cache_data, None)
        self.assertEqual(test_content.encode(), get_cache_data.content)


@override_settings(
    CACHE_MIDDLEWARE_KEY_PREFIX='settingsprefix',
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        },
    },
    LANGUAGES=(
        ('en', 'English'),
        ('es', 'Spanish'),
    ),
)
class CacheI18nTest(TestCase):

    def setUp(self):
        self.path = '/cache/test/'
        self.factory = RequestFactory()

    def tearDown(self):
        cache.clear()

    @override_settings(USE_I18N=True, USE_L10N=False, USE_TZ=False)
    def test_cache_key_i18n_translation(self):
        request = self.factory.get(self.path)
        lang = translation.get_language()
        response = HttpResponse()
        key = learn_cache_key(request, response)
        self.assertIn(lang, key, "Cache keys should include the language name when translation is active")
        key2 = get_cache_key(request)
        self.assertEqual(key, key2)

    def check_accept_language_vary(self, accept_language, vary, reference_key):
        request = self.factory.get(self.path)
        request.META['HTTP_ACCEPT_LANGUAGE'] = accept_language
        request.META['HTTP_ACCEPT_ENCODING'] = 'gzip;q=1.0, identity; q=0.5, *;q=0'
        response = HttpResponse()
        response['Vary'] = vary
        key = learn_cache_key(request, response)
        key2 = get_cache_key(request)
        self.assertEqual(key, reference_key)
        self.assertEqual(key2, reference_key)

    @override_settings(USE_I18N=True, USE_L10N=False, USE_TZ=False)
    def test_cache_key_i18n_translation_accept_language(self):
        lang = translation.get_language()
        self.assertEqual(lang, 'en')
        request = self.factory.get(self.path)
        request.META['HTTP_ACCEPT_ENCODING'] = 'gzip;q=1.0, identity; q=0.5, *;q=0'
        response = HttpResponse()
        response['Vary'] = 'accept-encoding'
        key = learn_cache_key(request, response)
        self.assertIn(lang, key, "Cache keys should include the language name when translation is active")
        self.check_accept_language_vary(
            'en-us',
            'cookie, accept-language, accept-encoding',
            key
        )
        self.check_accept_language_vary(
            'en-US',
            'cookie, accept-encoding, accept-language',
            key
        )
        self.check_accept_language_vary(
            'en-US,en;q=0.8',
            'accept-encoding, accept-language, cookie',
            key
        )
        self.check_accept_language_vary(
            'en-US,en;q=0.8,ko;q=0.6',
            'accept-language, cookie, accept-encoding',
            key
        )
        self.check_accept_language_vary(
            'ko-kr,ko;q=0.8,en-us;q=0.5,en;q=0.3 ',
            'accept-encoding, cookie, accept-language',
            key
        )
        self.check_accept_language_vary(
            'ko-KR,ko;q=0.8,en-US;q=0.6,en;q=0.4',
            'accept-language, accept-encoding, cookie',
            key
        )
        self.check_accept_language_vary(
            'ko;q=1.0,en;q=0.5',
            'cookie, accept-language, accept-encoding',
            key
        )
        self.check_accept_language_vary(
            'ko, en',
            'cookie, accept-encoding, accept-language',
            key
        )
        self.check_accept_language_vary(
            'ko-KR, en-US',
            'accept-encoding, accept-language, cookie',
            key
        )

    @override_settings(USE_I18N=False, USE_L10N=True, USE_TZ=False)
    def test_cache_key_i18n_formatting(self):
        request = self.factory.get(self.path)
        lang = translation.get_language()
        response = HttpResponse()
        key = learn_cache_key(request, response)
        self.assertIn(lang, key, "Cache keys should include the language name when formatting is active")
        key2 = get_cache_key(request)
        self.assertEqual(key, key2)

    @override_settings(USE_I18N=False, USE_L10N=False, USE_TZ=True)
    def test_cache_key_i18n_timezone(self):
        request = self.factory.get(self.path)
        # This is tightly coupled to the implementation,
        # but it's the most straightforward way to test the key.
        tz = force_text(timezone.get_current_timezone_name(), errors='ignore')
        tz = tz.encode('ascii', 'ignore').decode('ascii').replace(' ', '_')
        response = HttpResponse()
        key = learn_cache_key(request, response)
        self.assertIn(tz, key, "Cache keys should include the time zone name when time zones are active")
        key2 = get_cache_key(request)
        self.assertEqual(key, key2)

    @override_settings(USE_I18N=False, USE_L10N=False)
    def test_cache_key_no_i18n(self):
        request = self.factory.get(self.path)
        lang = translation.get_language()
        tz = force_text(timezone.get_current_timezone_name(), errors='ignore')
        tz = tz.encode('ascii', 'ignore').decode('ascii').replace(' ', '_')
        response = HttpResponse()
        key = learn_cache_key(request, response)
        self.assertNotIn(lang, key, "Cache keys shouldn't include the language name when i18n isn't active")
        self.assertNotIn(tz, key, "Cache keys shouldn't include the time zone name when i18n isn't active")

    @override_settings(USE_I18N=False, USE_L10N=False, USE_TZ=True)
    def test_cache_key_with_non_ascii_tzname(self):
        # Regression test for #17476
        class CustomTzName(timezone.UTC):
            name = ''

            def tzname(self, dt):
                return self.name

        request = self.factory.get(self.path)
        response = HttpResponse()
        with timezone.override(CustomTzName()):
            CustomTzName.name = 'Hora estándar de Argentina'.encode('UTF-8')  # UTF-8 string
            sanitized_name = 'Hora_estndar_de_Argentina'
            self.assertIn(sanitized_name, learn_cache_key(request, response),
                    "Cache keys should include the time zone name when time zones are active")

            CustomTzName.name = 'Hora estándar de Argentina'    # unicode
            sanitized_name = 'Hora_estndar_de_Argentina'
            self.assertIn(sanitized_name, learn_cache_key(request, response),
                    "Cache keys should include the time zone name when time zones are active")

    @override_settings(
        CACHE_MIDDLEWARE_KEY_PREFIX="test",
        CACHE_MIDDLEWARE_SECONDS=60,
        USE_ETAGS=True,
        USE_I18N=True,
    )
    def test_middleware(self):
        def set_cache(request, lang, msg):
            translation.activate(lang)
            response = HttpResponse()
            response.content = msg
            return UpdateCacheMiddleware().process_response(request, response)

        # cache with non empty request.GET
        request = self.factory.get(self.path, {'foo': 'bar', 'other': 'true'})
        request._cache_update_cache = True

        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        # first access, cache must return None
        self.assertEqual(get_cache_data, None)
        response = HttpResponse()
        content = 'Check for cache with QUERY_STRING'
        response.content = content
        UpdateCacheMiddleware().process_response(request, response)
        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        # cache must return content
        self.assertNotEqual(get_cache_data, None)
        self.assertEqual(get_cache_data.content, content.encode())
        # different QUERY_STRING, cache must be empty
        request = self.factory.get(self.path, {'foo': 'bar', 'somethingelse': 'true'})
        request._cache_update_cache = True
        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        self.assertEqual(get_cache_data, None)

        # i18n tests
        en_message = "Hello world!"
        es_message = "Hola mundo!"

        request = self.factory.get(self.path)
        request._cache_update_cache = True
        set_cache(request, 'en', en_message)
        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        # Check that we can recover the cache
        self.assertNotEqual(get_cache_data, None)
        self.assertEqual(get_cache_data.content, en_message.encode())
        # Check that we use etags
        self.assertTrue(get_cache_data.has_header('ETag'))
        # Check that we can disable etags
        with self.settings(USE_ETAGS=False):
            request._cache_update_cache = True
            set_cache(request, 'en', en_message)
            get_cache_data = FetchFromCacheMiddleware().process_request(request)
            self.assertFalse(get_cache_data.has_header('ETag'))
        # change the session language and set content
        request = self.factory.get(self.path)
        request._cache_update_cache = True
        set_cache(request, 'es', es_message)
        # change again the language
        translation.activate('en')
        # retrieve the content from cache
        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        self.assertEqual(get_cache_data.content, en_message.encode())
        # change again the language
        translation.activate('es')
        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        self.assertEqual(get_cache_data.content, es_message.encode())
        # reset the language
        translation.deactivate()

    @override_settings(
        CACHE_MIDDLEWARE_KEY_PREFIX="test",
        CACHE_MIDDLEWARE_SECONDS=60,
        USE_ETAGS=True,
    )
    def test_middleware_doesnt_cache_streaming_response(self):
        request = self.factory.get(self.path)
        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        self.assertIsNone(get_cache_data)

        # This test passes on Python < 3.3 even without the corresponding code
        # in UpdateCacheMiddleware, because pickling a StreamingHttpResponse
        # fails (http://bugs.python.org/issue14288). LocMemCache silently
        # swallows the exception and doesn't store the response in cache.
        content = ['Check for cache with streaming content.']
        response = StreamingHttpResponse(content)
        UpdateCacheMiddleware().process_response(request, response)

        get_cache_data = FetchFromCacheMiddleware().process_request(request)
        self.assertIsNone(get_cache_data)


@override_settings(
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
            'KEY_PREFIX': 'cacheprefix'
        },
    },
)
class PrefixedCacheI18nTest(CacheI18nTest):
    pass


def hello_world_view(request, value):
    return HttpResponse('Hello World %s' % value)


@override_settings(
    CACHE_MIDDLEWARE_ALIAS='other',
    CACHE_MIDDLEWARE_KEY_PREFIX='middlewareprefix',
    CACHE_MIDDLEWARE_SECONDS=30,
    CACHE_MIDDLEWARE_ANONYMOUS_ONLY=False,
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        },
        'other': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
            'LOCATION': 'other',
            'TIMEOUT': '1',
        },
    },
)
class CacheMiddlewareTest(IgnoreDeprecationWarningsMixin, TestCase):

    def setUp(self):
        super(CacheMiddlewareTest, self).setUp()
        self.factory = RequestFactory()
        self.default_cache = caches['default']
        self.other_cache = caches['other']

    def tearDown(self):
        self.default_cache.clear()
        self.other_cache.clear()
        super(CacheMiddlewareTest, self).tearDown()

    def test_constructor(self):
        """
        Ensure the constructor is correctly distinguishing between usage of CacheMiddleware as
        Middleware vs. usage of CacheMiddleware as view decorator and setting attributes
        appropriately.
        """
        # If no arguments are passed in construction, it's being used as middleware.
        middleware = CacheMiddleware()

        # Now test object attributes against values defined in setUp above
        self.assertEqual(middleware.cache_timeout, 30)
        self.assertEqual(middleware.key_prefix, 'middlewareprefix')
        self.assertEqual(middleware.cache_alias, 'other')
        self.assertEqual(middleware.cache_anonymous_only, False)

        # If arguments are being passed in construction, it's being used as a decorator.
        # First, test with "defaults":
        as_view_decorator = CacheMiddleware(cache_alias=None, key_prefix=None)

        self.assertEqual(as_view_decorator.cache_timeout, 30)  # Timeout value for 'default' cache, i.e. 30
        self.assertEqual(as_view_decorator.key_prefix, '')
        self.assertEqual(as_view_decorator.cache_alias, 'default')  # Value of DEFAULT_CACHE_ALIAS from django.core.cache
        self.assertEqual(as_view_decorator.cache_anonymous_only, False)

        # Next, test with custom values:
        as_view_decorator_with_custom = CacheMiddleware(cache_anonymous_only=True, cache_timeout=60, cache_alias='other', key_prefix='foo')

        self.assertEqual(as_view_decorator_with_custom.cache_timeout, 60)
        self.assertEqual(as_view_decorator_with_custom.key_prefix, 'foo')
        self.assertEqual(as_view_decorator_with_custom.cache_alias, 'other')
        self.assertEqual(as_view_decorator_with_custom.cache_anonymous_only, True)

    def test_middleware(self):
        middleware = CacheMiddleware()
        prefix_middleware = CacheMiddleware(key_prefix='prefix1')
        timeout_middleware = CacheMiddleware(cache_timeout=1)

        request = self.factory.get('/view/')

        # Put the request through the request middleware
        result = middleware.process_request(request)
        self.assertEqual(result, None)

        response = hello_world_view(request, '1')

        # Now put the response through the response middleware
        response = middleware.process_response(request, response)

        # Repeating the request should result in a cache hit
        result = middleware.process_request(request)
        self.assertNotEqual(result, None)
        self.assertEqual(result.content, b'Hello World 1')

        # The same request through a different middleware won't hit
        result = prefix_middleware.process_request(request)
        self.assertEqual(result, None)

        # The same request with a timeout _will_ hit
        result = timeout_middleware.process_request(request)
        self.assertNotEqual(result, None)
        self.assertEqual(result.content, b'Hello World 1')

    @override_settings(CACHE_MIDDLEWARE_ANONYMOUS_ONLY=True)
    def test_cache_middleware_anonymous_only_wont_cause_session_access(self):
        """ The cache middleware shouldn't cause a session access due to
        CACHE_MIDDLEWARE_ANONYMOUS_ONLY if nothing else has accessed the
        session. Refs 13283 """

        from django.contrib.sessions.middleware import SessionMiddleware
        from django.contrib.auth.middleware import AuthenticationMiddleware

        middleware = CacheMiddleware()
        session_middleware = SessionMiddleware()
        auth_middleware = AuthenticationMiddleware()

        request = self.factory.get('/view_anon/')

        # Put the request through the request middleware
        session_middleware.process_request(request)
        auth_middleware.process_request(request)
        result = middleware.process_request(request)
        self.assertEqual(result, None)

        response = hello_world_view(request, '1')

        # Now put the response through the response middleware
        session_middleware.process_response(request, response)
        response = middleware.process_response(request, response)

        self.assertEqual(request.session.accessed, False)

    @override_settings(CACHE_MIDDLEWARE_ANONYMOUS_ONLY=True)
    def test_cache_middleware_anonymous_only_with_cache_page(self):
        """CACHE_MIDDLEWARE_ANONYMOUS_ONLY should still be effective when used
        with the cache_page decorator: the response to a request from an
        authenticated user should not be cached."""

        request = self.factory.get('/view_anon/')

        class MockAuthenticatedUser(object):
            def is_authenticated(self):
                return True

        class MockAccessedSession(object):
            accessed = True

        request.user = MockAuthenticatedUser()
        request.session = MockAccessedSession()

        response = cache_page(60)(hello_world_view)(request, '1')

        self.assertFalse("Cache-Control" in response)

    def test_view_decorator(self):
        # decorate the same view with different cache decorators
        default_view = cache_page(3)(hello_world_view)
        default_with_prefix_view = cache_page(3, key_prefix='prefix1')(hello_world_view)

        explicit_default_view = cache_page(3, cache='default')(hello_world_view)
        explicit_default_with_prefix_view = cache_page(3, cache='default', key_prefix='prefix1')(hello_world_view)

        other_view = cache_page(1, cache='other')(hello_world_view)
        other_with_prefix_view = cache_page(1, cache='other', key_prefix='prefix2')(hello_world_view)

        request = self.factory.get('/view/')

        # Request the view once
        response = default_view(request, '1')
        self.assertEqual(response.content, b'Hello World 1')

        # Request again -- hit the cache
        response = default_view(request, '2')
        self.assertEqual(response.content, b'Hello World 1')

        # Requesting the same view with the explicit cache should yield the same result
        response = explicit_default_view(request, '3')
        self.assertEqual(response.content, b'Hello World 1')

        # Requesting with a prefix will hit a different cache key
        response = explicit_default_with_prefix_view(request, '4')
        self.assertEqual(response.content, b'Hello World 4')

        # Hitting the same view again gives a cache hit
        response = explicit_default_with_prefix_view(request, '5')
        self.assertEqual(response.content, b'Hello World 4')

        # And going back to the implicit cache will hit the same cache
        response = default_with_prefix_view(request, '6')
        self.assertEqual(response.content, b'Hello World 4')

        # Requesting from an alternate cache won't hit cache
        response = other_view(request, '7')
        self.assertEqual(response.content, b'Hello World 7')

        # But a repeated hit will hit cache
        response = other_view(request, '8')
        self.assertEqual(response.content, b'Hello World 7')

        # And prefixing the alternate cache yields yet another cache entry
        response = other_with_prefix_view(request, '9')
        self.assertEqual(response.content, b'Hello World 9')

        # But if we wait a couple of seconds...
        time.sleep(2)

        # ... the default cache will still hit
        caches['default']
        response = default_view(request, '11')
        self.assertEqual(response.content, b'Hello World 1')

        # ... the default cache with a prefix will still hit
        response = default_with_prefix_view(request, '12')
        self.assertEqual(response.content, b'Hello World 4')

        # ... the explicit default cache will still hit
        response = explicit_default_view(request, '13')
        self.assertEqual(response.content, b'Hello World 1')

        # ... the explicit default cache with a prefix will still hit
        response = explicit_default_with_prefix_view(request, '14')
        self.assertEqual(response.content, b'Hello World 4')

        # .. but a rapidly expiring cache won't hit
        response = other_view(request, '15')
        self.assertEqual(response.content, b'Hello World 15')

        # .. even if it has a prefix
        response = other_with_prefix_view(request, '16')
        self.assertEqual(response.content, b'Hello World 16')


@override_settings(
    CACHE_MIDDLEWARE_KEY_PREFIX='settingsprefix',
    CACHE_MIDDLEWARE_SECONDS=1,
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        },
    },
    USE_I18N=False,
)
class TestWithTemplateResponse(TestCase):
    """
    Tests various headers w/ TemplateResponse.

    Most are probably redundant since they manipulate the same object
    anyway but the Etag header is 'special' because it relies on the
    content being complete (which is not necessarily always the case
    with a TemplateResponse)
    """
    def setUp(self):
        self.path = '/cache/test/'
        self.factory = RequestFactory()

    def tearDown(self):
        cache.clear()

    def test_patch_vary_headers(self):
        headers = (
            # Initial vary, new headers, resulting vary.
            (None, ('Accept-Encoding',), 'Accept-Encoding'),
            ('Accept-Encoding', ('accept-encoding',), 'Accept-Encoding'),
            ('Accept-Encoding', ('ACCEPT-ENCODING',), 'Accept-Encoding'),
            ('Cookie', ('Accept-Encoding',), 'Cookie, Accept-Encoding'),
            ('Cookie, Accept-Encoding', ('Accept-Encoding',), 'Cookie, Accept-Encoding'),
            ('Cookie, Accept-Encoding', ('Accept-Encoding', 'cookie'), 'Cookie, Accept-Encoding'),
            (None, ('Accept-Encoding', 'COOKIE'), 'Accept-Encoding, COOKIE'),
            ('Cookie,     Accept-Encoding', ('Accept-Encoding', 'cookie'), 'Cookie, Accept-Encoding'),
            ('Cookie    ,     Accept-Encoding', ('Accept-Encoding', 'cookie'), 'Cookie, Accept-Encoding'),
        )
        for initial_vary, newheaders, resulting_vary in headers:
            response = TemplateResponse(HttpResponse(), Template("This is a test"))
            if initial_vary is not None:
                response['Vary'] = initial_vary
            patch_vary_headers(response, newheaders)
            self.assertEqual(response['Vary'], resulting_vary)

    def test_get_cache_key(self):
        request = self.factory.get(self.path)
        response = TemplateResponse(HttpResponse(), Template("This is a test"))
        key_prefix = 'localprefix'
        # Expect None if no headers have been set yet.
        self.assertEqual(get_cache_key(request), None)
        # Set headers to an empty list.
        learn_cache_key(request, response)
        self.assertEqual(get_cache_key(request), 'views.decorators.cache.cache_page.settingsprefix.GET.9fa0fd092afb73bdce204bb4f94d5804.d41d8cd98f00b204e9800998ecf8427e')
        # Verify that a specified key_prefix is taken into account.
        learn_cache_key(request, response, key_prefix=key_prefix)
        self.assertEqual(get_cache_key(request, key_prefix=key_prefix), 'views.decorators.cache.cache_page.localprefix.GET.9fa0fd092afb73bdce204bb4f94d5804.d41d8cd98f00b204e9800998ecf8427e')

    def test_get_cache_key_with_query(self):
        request = self.factory.get(self.path, {'test': 1})
        response = TemplateResponse(HttpResponse(), Template("This is a test"))
        # Expect None if no headers have been set yet.
        self.assertEqual(get_cache_key(request), None)
        # Set headers to an empty list.
        learn_cache_key(request, response)
        # Verify that the querystring is taken into account.
        self.assertEqual(get_cache_key(request), 'views.decorators.cache.cache_page.settingsprefix.GET.d11198ba31883732b0de5786a80cc12b.d41d8cd98f00b204e9800998ecf8427e')

    @override_settings(USE_ETAGS=False)
    def test_without_etag(self):
        response = TemplateResponse(HttpResponse(), Template("This is a test"))
        self.assertFalse(response.has_header('ETag'))
        patch_response_headers(response)
        self.assertFalse(response.has_header('ETag'))
        response = response.render()
        self.assertFalse(response.has_header('ETag'))

    @override_settings(USE_ETAGS=True)
    def test_with_etag(self):
        response = TemplateResponse(HttpResponse(), Template("This is a test"))
        self.assertFalse(response.has_header('ETag'))
        patch_response_headers(response)
        self.assertFalse(response.has_header('ETag'))
        response = response.render()
        self.assertTrue(response.has_header('ETag'))


class TestEtagWithAdmin(TestCase):
    # See https://code.djangoproject.com/ticket/16003
    urls = "admin_views.urls"

    def test_admin(self):
        with self.settings(USE_ETAGS=False):
            response = self.client.get('/test_admin/admin/')
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.has_header('ETag'))

        with self.settings(USE_ETAGS=True):
            response = self.client.get('/test_admin/admin/')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.has_header('ETag'))


class TestMakeTemplateFragmentKey(TestCase):
    def test_without_vary_on(self):
        key = make_template_fragment_key('a.fragment')
        self.assertEqual(key, 'template.cache.a.fragment.d41d8cd98f00b204e9800998ecf8427e')

    def test_with_one_vary_on(self):
        key = make_template_fragment_key('foo', ['abc'])
        self.assertEqual(key,
            'template.cache.foo.900150983cd24fb0d6963f7d28e17f72')

    def test_with_many_vary_on(self):
        key = make_template_fragment_key('bar', ['abc', 'def'])
        self.assertEqual(key,
            'template.cache.bar.4b35f12ab03cec09beec4c21b2d2fa88')

    def test_proper_escaping(self):
        key = make_template_fragment_key('spam', ['abc:def%'])
        self.assertEqual(key,
            'template.cache.spam.f27688177baec990cdf3fbd9d9c3f469')


class CacheHandlerTest(TestCase):
    def test_same_instance(self):
        """
        Attempting to retrieve the same alias should yield the same instance.
        """
        cache1 = caches['default']
        cache2 = caches['default']

        self.assertTrue(cache1 is cache2)

    def test_per_thread(self):
        """
        Requesting the same alias from separate threads should yield separate
        instances.
        """
        c = []

        def runner():
            c.append(caches['default'])

        for x in range(2):
            t = threading.Thread(target=runner)
            t.start()
            t.join()

        self.assertFalse(c[0] is c[1])
