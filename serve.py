#!/usr/bin/env python3
"""Tiny static file server with HTTP Range support (for audio seeking).

Serves the repo directory. Used for the private audition player page.
Usage: python3 serve.py [port]
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.isfile(path) or self.command != "GET":
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        try:
            fs = os.fstat(f.fileno())
            size = fs.st_size
            mtype = self.guess_type(path)
            rv = self.headers.get("Range")
            m = RANGE_RE.match(rv) if rv else None
            if m:
                s_s, e_s = m.group(1), m.group(2)
                if s_s == "":                       # suffix range: bytes=-N
                    length = min(int(e_s), size)
                    start, end = size - length, size - 1
                else:
                    start = int(s_s)
                    end = int(e_s) if e_s else size - 1
                    end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return None
                self.send_response(206)
                self.send_header("Content-Type", mtype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                f.seek(start)
                self._range_remaining = end - start + 1
                return _RangeFile(f, end - start + 1)
            self.send_response(200)
            self.send_header("Content-Type", mtype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return f
        except Exception:
            f.close()
            raise

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[%s] %s\n" % (self.address_string(), fmt % args))


class _RangeFile:
    """File wrapper that caps reads to the requested range length."""

    def __init__(self, f, remaining):
        self.f = f
        self.remaining = remaining

    def read(self, n=-1):
        if self.remaining <= 0:
            return b""
        n = self.remaining if (n < 0 or n > self.remaining) else n
        data = self.f.read(n)
        self.remaining -= len(data)
        return data


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    srv = ThreadingHTTPServer(("0.0.0.0", port), RangeHandler)
    print(f"Serving {ROOT} on http://0.0.0.0:{port} (Range supported)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
