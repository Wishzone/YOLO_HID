"""Generate a two-block HDMI 2.0 EDID for RK3588 HDMI input.

60/120 Hz use CTA timings. Higher modes use short, representable porches;
165/240 Hz match timings accepted by the source's existing gaming monitor.
The preferred mode stays at 60 Hz so reconnecting does not force a high rate.
"""
import argparse
import math
import struct
from pathlib import Path


def timing(refresh):
    if refresh in (60, 120):
        return dict(pixel_clock=14850 * (refresh // 60), h_active=1920,
                    h_blank=280, v_active=1080, v_blank=45,
                    h_sync_offset=88, h_sync_width=44,
                    v_sync_offset=4, v_sync_width=5, flags=0x1E)
    if refresh not in (144, 165, 180, 240):
        raise ValueError('Unsupported refresh rate')
    # Source-tested reduced-blanking HDMI timings (not CVT formula timings).
    h_blank = 150 if refresh == 240 else 160
    return dict(pixel_clock=math.ceil((1920 + h_blank) * 1125 * refresh / 10000),
                h_active=1920, h_blank=h_blank, v_active=1080, v_blank=45,
                h_sync_offset=48, h_sync_width=32,
                v_sync_offset=4, v_sync_width=5, flags=0x1A)


def create_dtd(t):
    pc = t['pixel_clock']
    ha, hb = t['h_active'], t['h_blank']
    va, vb = t['v_active'], t['v_blank']
    hso, hsw = t['h_sync_offset'], t['h_sync_width']
    vso, vsw = t['v_sync_offset'], t['v_sync_width']
    if not (0 < pc <= 60000 and 0 < ha <= 4095 and 0 < va <= 4095 and
            0 < hb <= 4095 and 0 < vb <= 4095 and 0 < hso <= 1023 and
            0 < hsw <= 1023 and 0 < vso <= 63 and 0 < vsw <= 63 and
            hso + hsw < hb and vso + vsw < vb):
        raise ValueError('Timing cannot be represented safely by an EDID DTD')
    dtd = bytearray(18)
    dtd[0:2] = struct.pack('<H', pc)
    dtd[2:5] = bytes((ha & 255, hb & 255, (ha >> 8) << 4 | hb >> 8))
    dtd[5:8] = bytes((va & 255, vb & 255, (va >> 8) << 4 | vb >> 8))
    dtd[8:12] = bytes((hso & 255, hsw & 255, (vso & 15) << 4 | vsw & 15,
                      (hso >> 8) << 6 | (hsw >> 8) << 4 | (vso >> 4) << 2 | vsw >> 4))
    dtd[12:15] = bytes((527 & 255, 296 & 255, (527 >> 8) << 4 | 296 >> 8))
    dtd[17] = t['flags']
    return dtd


def data_block(tag, payload):
    if not 0 < len(payload) <= 31:
        raise ValueError('Invalid CTA data block length')
    return bytes((tag << 5 | len(payload),)) + bytes(payload)


def checksum(block):
    block[127] = (-sum(block[:127])) & 255


def validate_edid(data):
    if len(data) < 128 or len(data) % 128 or data[:8] != b'\x00\xff\xff\xff\xff\xff\xff\x00':
        raise ValueError('Invalid EDID header or block size')
    if len(data) != (data[126] + 1) * 128:
        raise ValueError('EDID extension count does not match file size')
    for offset in range(0, len(data), 128):
        block = data[offset:offset+128]
        if sum(block) % 256:
            raise ValueError(f'Invalid checksum in block {offset // 128}')
        if offset == 0 or block[0] != 2:
            continue
        end = block[2]
        if not 4 <= end <= 127:
            raise ValueError('Invalid CTA DTD offset')
        index, previous_hdmi = 4, False
        while index < end:
            tag, size = block[index] >> 5, block[index] & 31
            if index + 1 + size > end:
                raise ValueError('CTA data block exceeds its declared boundary')
            payload = block[index+1:index+1+size]
            oui = payload[:3] if tag == 3 else b''
            if oui == b'\xd8\x5d\xc4':
                if size < 7 or not previous_hdmi:
                    raise ValueError('HDMI Forum block is truncated or is not directly after HDMI block')
                if payload[6] & 0xF0:
                    raise ValueError('This HDMI 2.0 profile must not advertise an FRL mode')
            previous_hdmi = oui == b'\x03\x0c\x00'
            index += 1 + size


def generate_edid():
    base = bytearray(128)
    base[:8] = b'\x00\xff\xff\xff\xff\xff\xff\x00'
    base[8:18] = b'\x49\x70\x88\x35\x01\x00\x00\x00\x01\x24'
    base[18:25] = b'\x01\x03\x80\x35\x1e\x78\x06'  # HDMI requires EDID 1.3; digital, sRGB, preferred
    base[25:35] = b'\xee\x91\xa3\x54\x4c\x99\x26\x0f\x50\x54'
    base[35:38] = b'\x21\x08\x00'  # VGA60, SVGA60, XGA60; no legacy 75-Hz-only fallback
    base[38:54] = b'\xd1\xc0' + b'\x01\x01' * 7
    base[54:72] = create_dtd(timing(60))
    base[72:90] = create_dtd(timing(144))
    base[90:108] = b'\x00\x00\x00\xfc\x00RK3588-Multi\n'
    # EDID 1.3 saturates the horizontal range at 255 kHz; individual DTDs
    # carry the higher horizontal frequency for 240 Hz explicitly.
    base[108:126] = b'\x00\x00\x00\xfd\x00\x32\xf0\x1e\xff\x3c\x00\x0a' + b' ' * 6
    base[126] = 1
    checksum(base)

    cta = bytearray(128)
    cta[0:4] = b'\x02\x03\x00\x81'  # Underscanned IT formats, RGB only, one native DTD
    blocks = (
        data_block(2, b'\x10\x3f') +  # CTA 1080p60/120; native 60 Hz is marked in the DTD list
        data_block(3, b'\x03\x0c\x00\x10\x00\x00\x44') +
        data_block(3, b'\xd8\x5d\xc4\x01\x78\x80\x00') +  # 600 MHz, SCDC; no FRL
        data_block(7, b'\x00\x4a')  # Selectable RGB range, underscanned IT/CE
    )
    index = 4 + len(blocks)
    cta[4:index] = blocks
    cta[2] = index
    for refresh in (60, 120, 165, 180, 240):
        cta[index:index+18] = create_dtd(timing(refresh))
        index += 18
    checksum(cta)
    result = bytes(base + cta)
    validate_edid(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('1080p_multi_hz.edid'))
    parser.add_argument('--check', type=Path, help='Validate an existing EDID without writing')
    args = parser.parse_args()
    if args.check:
        validate_edid(args.check.read_bytes())
        print(f'EDID structure/checksums valid: {args.check}')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(generate_edid())
        print(f'Generated {args.output}: 1080p 60/120/144/165/180/240 Hz (advertised modes)')
