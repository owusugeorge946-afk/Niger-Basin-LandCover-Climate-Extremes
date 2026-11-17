import json
import tempfile
import unittest
from pathlib import Path
from run import preflight, MANIFEST

class ManifestTests(unittest.TestCase):
    def test_expected_counts(self):
        self.assertEqual(MANIFEST['expected_outputs']['figures'],9)
        self.assertEqual(MANIFEST['expected_outputs']['tables'],12)
        self.assertEqual(MANIFEST['main_grid_cells'],2661)

    def test_missing_inputs_fail_clearly(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p)
            ok,missing=preflight(root,root/'Niger_Basin_Reviewer1_Revision')
            self.assertFalse(ok)
            self.assertGreater(len(missing),10)

    def test_empty_placeholder_files_pass_presence_check(self):
        # This tests manifest resolution only; no scientific validation.
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);base=root/'Niger_Basin_Reviewer1_Revision'
            for subdir, files in MANIFEST['required_files'].items():
                for fname in files:
                    x=base/subdir/fname;x.parent.mkdir(parents=True,exist_ok=True);x.write_text('placeholder')
            for rel in MANIFEST['external_files']:
                x=root/rel;x.parent.mkdir(parents=True,exist_ok=True);x.write_text('placeholder')
            ok,missing=preflight(root,base)
            self.assertTrue(ok)
            self.assertFalse(missing)

if __name__=='__main__':unittest.main()
