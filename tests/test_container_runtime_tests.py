import unittest

from scripts import run_container_tests as runtime


class ContainerRuntimeTestSelectionTests(unittest.TestCase):
    def test_source_repository_tests_are_explicitly_excluded(self):
        class RuntimeCase(unittest.TestCase):
            def test_runtime(self):
                pass

        class SourceCase(unittest.TestCase):
            requires_source_repository = True

            def test_evidence(self):
                pass

        suite = unittest.TestSuite((RuntimeCase("test_runtime"), SourceCase("test_evidence")))
        selected, excluded = runtime.runtime_suite(suite)
        self.assertEqual([test.id() for test in runtime.iter_test_cases(selected)],
                         [RuntimeCase("test_runtime").id()])
        self.assertEqual(excluded, [SourceCase("test_evidence").id()])


if __name__ == "__main__":
    unittest.main()
