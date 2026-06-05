import subprocess
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]


class ForgeWrapperTests(unittest.TestCase):
    def test_top_level_help_exits_without_error(self) -> None:
        result = subprocess.run(
            [str(REPO_DIR / "forge"), "--help"],
            cwd=REPO_DIR,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout)

    def test_serve_help_does_not_launch_server(self) -> None:
        result = subprocess.run(
            [str(REPO_DIR / "forge"), "serve", "qwen36_27b", "base", "--help"],
            cwd=REPO_DIR,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("./forge serve <family> [variant]", result.stdout)
        self.assertNotIn("Starting", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
