import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('gen_edid', Path(__file__).parents[1] / 'tools/gen_edid.py')
edid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edid)


class EdidTests(unittest.TestCase):
    def test_modes_and_field_limits(self):
        for refresh in (60, 120, 144, 165, 180, 240):
            block = edid.create_dtd(edid.timing(refresh))
            pc = int.from_bytes(block[:2], 'little') * 10000
            ha = block[2] | (block[4] >> 4) << 8
            hb = block[3] | (block[4] & 15) << 8
            va = block[5] | (block[7] >> 4) << 8
            vb = block[6] | (block[7] & 15) << 8
            vso = (block[10] >> 4) | ((block[11] >> 2) & 3) << 4
            self.assertEqual((ha, va), (1920, 1080))
            self.assertLessEqual(pc, 600000000)
            self.assertEqual(vso, edid.timing(refresh)['v_sync_offset'])
            self.assertAlmostEqual(pc / (ha+hb) / (va+vb), refresh, delta=0.01)

    def test_blocks_and_checksums(self):
        data = edid.generate_edid()
        self.assertEqual(len(data), 256)
        edid.validate_edid(data)
        corrupted = bytearray(data)
        corrupted[240] ^= 1
        with self.assertRaises(ValueError):
            edid.validate_edid(corrupted)

    def test_truncated_hdmi_forum_block(self):
        data = bytearray(edid.generate_edid())
        index = data.index(b'\x67\xd8\x5d\xc4', 128)
        data[index] = 0x66
        extension = data[128:]
        edid.checksum(extension)
        data[128:] = extension
        with self.assertRaisesRegex(ValueError, 'truncated'):
            edid.validate_edid(data)

    def test_oversized_dtd_porch_rejected(self):
        bad = edid.timing(240)
        bad['v_sync_offset'] = 121
        with self.assertRaises(ValueError):
            edid.create_dtd(bad)


if __name__ == '__main__':
    unittest.main()
