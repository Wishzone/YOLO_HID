#!/usr/bin/env python3
"""Local UI for the shared C++ aim simulator; standard library only."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import subprocess
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "aim_simulator"
OPTIONS = frozenset((
    "width", "height", "frames", "frame-ms", "feedback-ms", "hid-interval-ms",
    "offset-x", "offset-y", "box-width", "box-height", "vx", "vy", "gain-x", "gain-y",
    "kp", "kd", "sensitivity", "deadzone", "hid-smoothing", "prediction-ms",
    "aim-ratio", "head", "settle-frames", "tolerance",
))


def simulate_query(query):
    if len(query) > 8192:
        raise ValueError("Too many parameters")
    params = parse_qs(query, keep_blank_values=True, max_num_fields=30)
    args = [str(BINARY)]
    output_format = "json"
    for key, values in params.items():
        if len(values) != 1:
            raise ValueError("Repeated option: " + key)
        if key == "format":
            output_format = values[0]
            if output_format not in ("json", "csv"):
                raise ValueError("Invalid format")
        elif key in OPTIONS:
            args.extend(["--" + key, values[0]])
        else:
            raise ValueError("Unknown option: " + key)
    args.extend(["--format", output_format])
    result = subprocess.run(args, capture_output=True, timeout=15, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout, output_format


class SimulatorHandler(BaseHTTPRequestHandler):
    def send_body(self, status, content_type, body, attachment=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if attachment:
            self.send_header("Content-Disposition", 'attachment; filename="' + attachment + '"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        request = urlsplit(self.path)
        try:
            if request.path in ("/", "/aim_simulator.html"):
                self.send_body(200, "text/html; charset=utf-8", (ROOT / "tools" / "aim_simulator.html").read_bytes())
            elif request.path == "/api/simulate":
                body, output_format = simulate_query(request.query)
                csv = output_format == "csv"
                self.send_body(200, "text/csv; charset=utf-8" if csv else "application/json",
                               body, "aim_trace.csv" if csv else None)
            else:
                self.send_body(404, "text/plain; charset=utf-8", b"Not found")
        except (ValueError, subprocess.TimeoutExpired) as error:
            self.send_body(400, "text/plain; charset=utf-8", str(error).encode("utf-8"))
        except OSError as error:
            self.send_body(500, "text/plain; charset=utf-8", str(error).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to view from another computer")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    if not BINARY.is_file():
        parser.error("Run make build/aim_simulator first")
    server = HTTPServer((args.host, args.port), SimulatorHandler)
    print(f"Aim simulator: http://{args.host}:{args.port}/ (offline simulation, no hardware output)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
