from scapy.all import rdpcap

pcap_file = "wlan0.pcap"
packets = rdpcap(pcap_file)

found_csi = False
for i, pkt in enumerate(packets):
    raw = bytes(pkt)
    # Nexmon CSI 帧的固定长度头一般在 radiotap 后面出现
    if b'NEXMON' in raw or len(raw) > 1000:  # 粗略检测
        print(f"[DEBUG] Possible CSI data in packet {i}, length={len(raw)}")
        found_csi = True
        break

if not found_csi:
    print("[ERR] No CSI data found in this pcap (maybe wrong nexutil -s parameter?)")
