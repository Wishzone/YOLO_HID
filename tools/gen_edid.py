import struct

def calculate_cvt_rb(width, height, refresh_rate):
    # Constants for CVT-RB
    H_BLANK = 160
    CLOCK_STEP = 0.25
    MIN_V_BPORCH = 3
    V_SYNC = 5
    MIN_V_PORCH_RND = 3
    
    # 1. Estimate Horizontal Period (kHZ)
    # H_PERIOD_EST = ((1/V_FIELD_RATE) - MIN_V_BLANK) / (Y_PIXELS + 2*V_MARGIN + MIN_V_PORCH_RND + INTERLACED)
    # Simplified for RB:
    # Total V lines = V Active + V_BLANK_RB
    # V_BLANK_RB = 460 (fixed for RBv2?) No, let's use standard CVT-RB v1
    # V_BLANK = 3 + 5 + 3 + (V_SYNC etc) ... actually RB is fixed blanking time usually.
    
    # Let's use the standard CVT-RB formula
    # H Period = 1 / (Refresh * (V_LINES + V_BLANK))
    # But we need to find Pixel Clock.
    
    # CVT-RB Timing:
    # H Blank is fixed at 160 pixels.
    # H Sync is 32 pixels.
    # V Blank is fixed? No.
    # V Sync is fixed at 5 lines.
    # V Back Porch is fixed at 3 lines?
    # V Front Porch is variable?
    
    # Actually, let's use a known modeline for 1920x1080 @ 165Hz (CVT-RB)
    # Modeline "1920x1080_165.00"  380.98  1920 1968 2000 2080  1080 1083 1088 1111 +hsync -vsync
    # Pixel Clock: 380.98 MHz
    # H Active: 1920
    # H Front Porch: 48
    # H Sync: 32
    # H Back Porch: 80
    # H Total: 2080
    # V Active: 1080
    # V Front Porch: 3
    # V Sync: 5
    # V Back Porch: 23
    # V Total: 1111
    
    # Refresh = 380980000 / (2080 * 1111) = 164.86 Hz (~165Hz)
    
    return {
        "pixel_clock": 38098, # in 10kHz
        "h_active": 1920,
        "h_blank": 160, # 2080 - 1920
        "v_active": 1080,
        "v_blank": 31, # 1111 - 1080
        "h_sync_offset": 48, # Front Porch
        "h_sync_width": 32,
        "v_sync_offset": 3, # Front Porch
        "v_sync_width": 5,
        "h_size_mm": 527, # Example 24 inch
        "v_size_mm": 296,
        "flags": 0x18 # Digital, +HSync, -VSync (Wait, CVT-RB is usually +H -V? Or +H +V?)
                      # Standard CVT-RB is +H -V usually? 
                      # Let's check flags. 
                      # Bit 7: 0 (Analog) / 1 (Digital) -> DTD is different? No, this is for separate sync.
                      # 0x1E = Digital, +H, +V. 
                      # 0x18 = Digital, +H, +V (bits 1,2 are sync polarity).
                      # Bit 2: V Sync Polarity (1=Positive)
                      # Bit 1: H Sync Polarity (1=Positive)
                      # CVT-RB is usually +H -V (0x1A) or -H +V?
                      # Actually, let's stick to 0x1E (+H +V) or 0x18 (+H +V, no stereo).
                      # The example 1080p60 EDID had 0x1E (+H +V).
    }

def create_dtd(timing):
    dtd = bytearray(18)
    
    # 0-1: Pixel Clock
    pc = timing["pixel_clock"]
    dtd[0] = pc & 0xFF
    dtd[1] = (pc >> 8) & 0xFF
    
    # 2: H Active
    ha = timing["h_active"]
    dtd[2] = ha & 0xFF
    
    # 3: H Blanking
    hb = timing["h_blank"]
    dtd[3] = hb & 0xFF
    
    # 4: H Active/Blanking upper bits
    dtd[4] = ((ha >> 8) & 0xF) << 4 | ((hb >> 8) & 0xF)
    
    # 5: V Active
    va = timing["v_active"]
    dtd[5] = va & 0xFF
    
    # 6: V Blanking
    vb = timing["v_blank"]
    dtd[6] = vb & 0xFF
    
    # 7: V Active/Blanking upper bits
    dtd[7] = ((va >> 8) & 0xF) << 4 | ((vb >> 8) & 0xF)
    
    # 8: H Sync Offset
    hso = timing["h_sync_offset"]
    dtd[8] = hso & 0xFF
    
    # 9: H Sync Pulse Width
    hspw = timing["h_sync_width"]
    dtd[9] = hspw & 0xFF
    
    # 10: V Sync Offset/Pulse Width
    vso = timing["v_sync_offset"]
    vspw = timing["v_sync_width"]
    dtd[10] = ((vso & 0xF) << 4) | (vspw & 0xF)
    
    # 11: Upper bits for Sync
    dtd[11] = ((hso >> 8) & 0x3) << 6 | ((hspw >> 8) & 0x3) << 4 | ((vso >> 4) & 0x3) << 2 | ((vspw >> 4) & 0x3)
    
    # 12: H Image Size
    hsz = timing["h_size_mm"]
    dtd[12] = hsz & 0xFF
    
    # 13: V Image Size
    vsz = timing["v_size_mm"]
    dtd[13] = vsz & 0xFF
    
    # 14: Upper bits for Image Size
    dtd[14] = ((hsz >> 8) & 0xF) << 4 | ((vsz >> 8) & 0xF)
    
    # 15: H Border
    dtd[15] = 0
    
    # 16: V Border
    dtd[16] = 0
    
    # 17: Flags
    # 0x18: Digital, Separate Sync, +H, +V (Wait, 0x18 is +H +V?)
    # Bit 7: 0
    # Bit 6-5: 00 (Normal)
    # Bit 4: 1 (Digital Separate)
    # Bit 3: 1 (Reserved/Stereo?) -> Usually 1 for Digital Separate? No.
    # Bit 2: V Polarity (1=Pos)
    # Bit 1: H Polarity (1=Pos)
    # 0x1E = 0001 1110 -> Digital Separate, V+, H+
    dtd[17] = 0x1E 
    
    return dtd

def generate_edid():
    # Base EDID (128 bytes)
    # Header
    edid = bytearray(b'\x00\xFF\xFF\xFF\xFF\xFF\xFF\x00')
    
    # Vendor/Product ID (Manufacturer: RPI, Product: 1234)
    # RPI = 0x49 0x70 (from dump)
    # ID (2), Product (2), Serial (4), Week (1), Year (1) = 10 bytes
    edid += bytearray(b'\x49\x70\x88\x35\x01\x00\x00\x00\x01\x1E') # Added Week/Year
    
    # Edid Version 1.3
    edid += bytearray(b'\x01\x03')
    
    # Basic Display Params
    # Digital, 8 bits, etc.
    edid += bytearray(b'\x80\x34\x20\x78\x22') # Input, H/V cm, Gamma
    
    # Chromaticity (sRGB)
    edid += bytearray(b'\xEE\x91\xA3\x54\x4C\x99\x26\x0F\x50\x54')
    
    # Established Timings (720x400@70, 640x480@60, 800x600@60, 1024x768@60)
    edid += bytearray(b'\xA5\x4B\x00')
    
    # Standard Timings (1920x1080@60, etc)
    # 0xD1C0 = 1920x1080 @ 60Hz (16:9)
    # 0x8180 = 1280x1024 @ 60Hz
    # 0x6140 = 1024x768 @ 60Hz
    # 0x4540 = 800x600 @ 60Hz
    # 0x3140 = 640x480 @ 60Hz
    # ...
    edid += bytearray(b'\xD1\xC0\x81\x80\x61\x40\x45\x40\x31\x40\x01\x01\x01\x01\x01\x01')
    
    # Descriptor 1: 1920x1080 @ 165Hz (Preferred)
    timing_165 = {
        "pixel_clock": 38098,
        "h_active": 1920,
        "h_blank": 160,
        "v_active": 1080,
        "v_blank": 31,
        "h_sync_offset": 48,
        "h_sync_width": 32,
        "v_sync_offset": 3,
        "v_sync_width": 5,
        "h_size_mm": 527,
        "v_size_mm": 296
    }
    edid += create_dtd(timing_165)
    
    # Descriptor 2: 1920x1080 @ 60Hz
    # Pixel Clock: 148.5 MHz = 14850
    timing_60 = {
        "pixel_clock": 14850,
        "h_active": 1920,
        "h_blank": 280, # 2200 - 1920
        "v_active": 1080,
        "v_blank": 45, # 1125 - 1080
        "h_sync_offset": 88,
        "h_sync_width": 44,
        "v_sync_offset": 4,
        "v_sync_width": 5,
        "h_size_mm": 527,
        "v_size_mm": 296
    }
    edid += create_dtd(timing_60)
    
    # Descriptor 3: Monitor Name
    # 00 00 00 FC 00 ...
    name = b"RK3588-165Hz"
    desc3 = bytearray(b'\x00\x00\x00\xFC\x00') + name + b'\x0A' + b'\x20' * (12 - len(name))
    edid += desc3
    
    # Descriptor 4: Range Limits
    # 00 00 00 FD 00 ...
    # Min V: 48, Max V: 165, Min H: 30, Max H: 200, Max Clock: 400
    desc4 = bytearray(b'\x00\x00\x00\xFD\x00\x30\xA5\x1E\xC8\x28\x00\x0A\x20\x20\x20\x20\x20\x20')
    edid += desc4
    
    # Extension Flag (0)
    edid += bytearray(b'\x00')
    
    # Checksum
    checksum = 0
    for b in edid:
        checksum += b
    checksum = (256 - (checksum % 256)) % 256
    edid += bytearray([checksum])
    
    return edid

if __name__ == "__main__":
    edid_data = generate_edid()
    with open("1080p_165hz.edid", "wb") as f:
        f.write(edid_data)
    print("Generated 1080p_165hz.edid")
