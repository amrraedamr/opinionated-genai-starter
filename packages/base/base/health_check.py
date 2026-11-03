"""Container healthcheck probe."""

import os
import sys
import urllib.request


def main() -> int:
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = os.environ.get("API_PORT", "8000")
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/livez"
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            return 0 if response.status == 200 else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
