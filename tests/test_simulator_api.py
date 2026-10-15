import csv
from http.server import HTTPServer
import importlib.util
import io
import json
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aim_simulator", ROOT / "tools" / "aim_simulator.py")
SIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIM)


class SimulatorAPI(unittest.TestCase):
    def test_default_trace_and_csv(self):
        raw, kind = SIM.simulate_query("")
        data = json.loads(raw)
        self.assertEqual(kind, "json")
        self.assertEqual(data["options"]["frame-ms"], 20)
        self.assertEqual(len(data["trace"]), 151)
        self.assertEqual(data["trace"][-1]["time_ms"], 3000)
        self.assertEqual(data["summary"]["first_entry_frame"], 7)
        self.assertEqual(data["summary"]["first_aim_frame"], 17)
        self.assertEqual(sum(row["reports"] for row in data["trace"]), data["summary"]["reports"])
        raw_csv, kind = SIM.simulate_query("format=csv")
        rows = list(csv.DictReader(io.StringIO(raw_csv.decode())))
        self.assertEqual(kind, "csv")
        self.assertEqual(len(rows), 151)
        self.assertEqual(float(rows[1]["hid_dx"]), data["trace"][1]["hid_dx"])
        # JSON export can be sent straight back to the calculation endpoint.
        from urllib.parse import urlencode
        rerun, _ = SIM.simulate_query(urlencode(data["options"]))
        self.assertEqual(json.loads(rerun), data)

    def test_invalid_parameters(self):
        for query in ("kp=nan", "kp=inf", "kp=-1", "head=2", "frames=1.5", "frames=0",
                      "frame-ms=0", "gain-x=0", "prediction-ms=41", "format=html",
                      "kp=1&kp=2", "unknown=1", "kp=", "kp=1%3Btouch%20no",
                      "frames=999999999999999999999", "head=-1"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                SIM.simulate_query(query)

    def test_negative_velocity_and_short_run(self):
        raw, _ = SIM.simulate_query("offset-x=-300&vx=-1e2&frames=1")
        data = json.loads(raw)
        self.assertLess(data["trace"][1]["camera_x"], 0)
        self.assertEqual(data["summary"]["settled_frame"], -1)

    def test_http_routes(self):
        server = HTTPServer(("127.0.0.1", 0), SIM.SimulatorHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/", timeout=5) as response:
                self.assertIn("固定中心准星", response.read().decode())
            with urlopen(base + "/api/simulate?frames=2&format=csv", timeout=5) as response:
                self.assertIn("aim_trace.csv", response.headers["Content-Disposition"])
                self.assertEqual(len(list(csv.DictReader(io.StringIO(response.read().decode())))), 3)
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + "/api/simulate?kp=nan", timeout=5)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + "/missing", timeout=5)
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
