import subprocess
import sys
from pathlib import Path

import django
from django.test import SimpleTestCase


class DeprecationTests(SimpleTestCase):
    DEPRECATION_MESSAGE = (
        b'RemovedInDjango40Warning: django-admin.py is deprecated in favor of '
        b'django-admin.'
    )

    def _run_test(self, args):
        p = subprocess.run(
            [sys.executable, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return p.stdout, p.stderr

    def test_django_admin_py_deprecated(self):
        django_admin_py = Path(django.__file__).parent / 'bin' / 'django-admin.py'
        _, err = self._run_test(['-Wd', django_admin_py, '--version'])
        self.assertIn(self.DEPRECATION_MESSAGE, err)

    def test_main_not_deprecated(self):
        _, err = self._run_test(['-Wd', '-m', 'django', '--version'])
        self.assertNotIn(self.DEPRECATION_MESSAGE, err)

    def test_django_admin_py_equivalent_main(self):
        django_admin_py = Path(django.__file__).parent / 'bin' / 'django-admin.py'
        django_admin_py_out, _ = self._run_test([django_admin_py, '--version'])
        django_out, _ = self._run_test(['-m', 'django', '--version'])
        self.assertEqual(django_admin_py_out, django_out)
