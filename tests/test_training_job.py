"""Check status publishing while a Windows reader temporarily holds the file."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import training_job


@unittest.skipUnless(os.name == "nt", "Windows file-sharing regression")
class StatusFileTests(unittest.TestCase):
    def test_publish_waits_for_reader_to_release_file(self):
        kernel = ctypes.windll.kernel32
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        training_job.CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=training_job.CACHE) as directory:
            path = Path(directory) / "status.json"
            path.write_text('{"state":"old"}', encoding="utf-8")
            # Permit reads/writes but deny deletion, as Windows readers can do.
            handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            release = threading.Timer(0.25, lambda: kernel.CloseHandle(handle))
            release.start()
            try:
                with patch.object(training_job, "STATUS", path):
                    training_job.save_status({"state": "complete"})
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["state"], "complete")
                self.assertIn("updated_utc", saved)
            finally:
                release.join()


if __name__ == "__main__":
    unittest.main()
