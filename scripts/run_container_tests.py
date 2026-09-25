#!/usr/bin/env python3
"""Run tests compatible with the deliberately stripped runtime image.

Historical qualification evidence is source-repository content rather than a
runtime-image dependency.  Tests marked ``requires_source_repository`` run in
the workflow against the checked-out source; this runner reports and excludes
only those tests from image validation.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


SOURCE_REPOSITORY_MARKER = "requires_source_repository"


def iter_test_cases(suite: unittest.TestSuite):
    """Yield individual test cases from a potentially nested suite."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_test_cases(item)
        else:
            yield item


def requires_source_repository(test: unittest.TestCase) -> bool:
    method = getattr(test, test._testMethodName)
    return any(getattr(subject, SOURCE_REPOSITORY_MARKER, False)
               for subject in (test, test.__class__, method))


def runtime_suite(discovered: unittest.TestSuite) -> tuple[unittest.TestSuite, list[str]]:
    """Return executable runtime tests and explicitly reported source-only IDs."""
    selected = unittest.TestSuite()
    excluded: list[str] = []
    for test in iter_test_cases(discovered):
        if requires_source_repository(test):
            excluded.append(test.id())
        else:
            selected.addTest(test)
    return selected, excluded


def discover_runtime_suite(repository: Path) -> tuple[unittest.TestSuite, list[str]]:
    tests = repository / "tests"
    # Existing tests import some peer fixtures as top-level modules; this
    # mirrors ``python -m unittest discover -s tests``.
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(tests))
    loader = unittest.defaultTestLoader
    discovered = loader.discover(str(tests), pattern="test_*.py")
    return runtime_suite(discovered)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    suite, excluded = discover_runtime_suite(repository)
    if excluded:
        print("Source-repository-only tests excluded from runtime image validation:")
        for test_id in excluded:
            print(f"  {test_id}")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
