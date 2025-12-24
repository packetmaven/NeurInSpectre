#!/usr/bin/env python3
"""
Generate a small, realistic-looking PCAP with timing (IPD) patterns that are
useful for testing NeurInSpectre's DeMarking / watermarking-evasion detector.

Why this exists
---------------
NeurInSpectre's `evasion-detect --network-data <pcap>` is driven primarily by
inter-packet delays (IPDs). The ESORICS/ESORICSW 2024 workshop chapter
"Generating Traffic-Level Adversarial Examples from Feature-Level Specifications"
describes bridging **feature-level timing specifications** back into valid
traffic sequences (PCAP) via constrained generation.

This script implements a *minimal* version of that idea: we define a timing
spec (IPD series) and materialize it into a classic libpcap file with benign
packet bytes but adversarial timing.

Reference:
  - https://link.springer.com/chapter/10.1007/978-3-031-82362-6_8

Usage
-----
python -m neurinspectre.tools.generate_demarking_pcap --out network_flows.pcap

Then:
neurinspectre evasion-detect real_attention.npy \
  --network-data network_flows.pcap \
  --detector-type demarking \
  --threshold 0.6 \
  --output-dir _cli_runs/demarking
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

try:
    # Optional: used to "double-check" that the generated timing spec trips the detector.
    from neurinspectre.security.evasion_detection import DeMarkingDefenseDetector  # type: ignore
except Exception:  # pragma: no cover
    DeMarkingDefenseDetector = None  # type: ignore


def _build_dummy_eth_ipv4_udp_frame(payload_len: int = 32) -> bytes:
    """Return a minimal Ethernet+IPv4+UDP frame (bytes).

    Packet bytes are not parsed by the DeMarking detector; only PCAP timestamps
    matter. Still, we generate plausible frames for tooling compatibility.
    """
    # Ethernet header (14 bytes)
    dst = b"\xaa\xbb\xcc\xdd\xee\xff"
    src = b"\x11\x22\x33\x44\x55\x66"
    eth_type = b"\x08\x00"  # IPv4
    eth = dst + src + eth_type

    # IPv4 header (20 bytes, no options)
    version_ihl = 0x45
    tos = 0
    total_length = 20 + 8 + payload_len
    identification = 0
    flags_frag = 0
    ttl = 64
    proto = 17  # UDP
    checksum = 0  # ignore checksum correctness for synthetic traffic
    src_ip = b"\x0a\x00\x00\x01"  # 10.0.0.1
    dst_ip = b"\x0a\x00\x00\x02"  # 10.0.0.2
    ip = struct.pack("!BBHHHBBH4s4s", version_ihl, tos, total_length, identification, flags_frag, ttl, proto, checksum, src_ip, dst_ip)

    # UDP header (8 bytes)
    src_port = 44444
    dst_port = 55555
    udp_length = 8 + payload_len
    udp_checksum = 0
    udp = struct.pack("!HHHH", src_port, dst_port, udp_length, udp_checksum)

    payload = b"x" * payload_len
    return eth + ip + udp + payload


def _load_pcap_ipds(p: Path) -> np.ndarray:
    """Load inter-packet delays (seconds) from a classic libpcap (.pcap) file."""
    data = p.read_bytes()
    if len(data) < 24:
        raise ValueError(f"PCAP too small: {p}")

    magic = data[:4]
    # Classic pcap magic numbers:
    # 0xa1b2c3d4 (microsecond, big-endian), 0xd4c3b2a1 (microsecond, little-endian)
    # 0xa1b23c4d (nanosecond, big-endian), 0x4d3cb2a1 (nanosecond, little-endian)
    if magic == b"\xa1\xb2\xc3\xd4":
        endian = ">"
        ts_scale = 1e-6
    elif magic == b"\xd4\xc3\xb2\xa1":
        endian = "<"
        ts_scale = 1e-6
    elif magic == b"\xa1\xb2\x3c\x4d":
        endian = ">"
        ts_scale = 1e-9
    elif magic == b"\x4d\x3c\xb2\xa1":
        endian = "<"
        ts_scale = 1e-9
    else:
        raise ValueError(f"Unsupported PCAP magic {magic!r} in {p} (pcapng not supported here)")

    off = 24
    ph_fmt = endian + "IIII"  # ts_sec, ts_subsec, incl_len, orig_len
    ph_sz = struct.calcsize(ph_fmt)

    ts: list[float] = []
    while off + ph_sz <= len(data):
        ts_sec, ts_sub, incl_len, _orig_len = struct.unpack_from(ph_fmt, data, off)
        off += ph_sz
        if incl_len < 0 or off + incl_len > len(data):
            break
        off += incl_len
        ts.append(float(ts_sec) + float(ts_sub) * ts_scale)

    if len(ts) < 2:
        return np.zeros((0,), dtype=np.float64)

    ts_arr = np.asarray(ts, dtype=np.float64)
    ipd = np.diff(ts_arr)
    ipd = np.nan_to_num(ipd, nan=0.0, posinf=0.0, neginf=0.0)
    ipd[ipd < 0.0] = 0.0
    return ipd


def _timing_spec_ipds(n_packets: int, *, seed: int) -> np.ndarray:
    """Create an IPD series (seconds) with a watermark/evasion-style signature.

    Design intent:
    - **Mode collapse** (few distinct delay values) -> trips GAN/mode-collapse heuristics
    - **High lag correlation** (quantized, correlated process) -> trips temporal correlation heuristics
    - **Non-stationarity** (occasional jumps) -> reduces chunk consistency, raising score
    - **Timing-only adversary**: payload stays benign; only timestamps encode the behavior
    """
    rng = np.random.default_rng(int(seed))

    # Quantized, highly persistent AR(1) with occasional "jumps".
    levels = np.linspace(0.010, 0.090, 6)  # seconds
    mu = 0.058
    phi = 0.988
    sigma = 0.0014
    p_jump = 0.013

    ipd = np.empty(n_packets - 1, dtype=np.float64)
    state = float(mu)
    for i in range(ipd.size):
        state = float(mu) + float(phi) * (state - float(mu)) + float(rng.normal(0.0, sigma))
        if float(rng.random()) < float(p_jump):
            state = float(rng.choice(levels))
        j = int(np.argmin(np.abs(levels - state)))
        state = float(levels[j])
        ipd[i] = state

    ipd = np.clip(ipd, 0.001, 0.250)

    return ipd.astype(np.float64, copy=False)

def _pcap_microsecond_quantize_ipds(ipds_s: np.ndarray, *, start_epoch_s: float = 1_700_000_000.0) -> np.ndarray:
    """Simulate PCAP microsecond timestamp quantization and return the resulting IPDs.

    PCAP stores per-packet timestamps at microsecond resolution (in this script),
    so the IPDs seen by the CLI are the diffs of these quantized timestamps.
    """
    ipds_s = np.asarray(ipds_s, dtype=np.float64).reshape(-1)
    ts = np.empty(ipds_s.size + 1, dtype=np.float64)
    ts[0] = float(start_epoch_s)
    ts[1:] = ts[0] + np.cumsum(ipds_s)

    q = np.empty_like(ts)
    for i, t in enumerate(ts):
        sec = int(np.floor(t))
        usec = int(round((t - sec) * 1e6))
        if usec >= 1_000_000:
            sec += 1
            usec -= 1_000_000
        q[i] = sec + usec / 1e6

    ipd_q = np.diff(q)
    ipd_q = np.nan_to_num(ipd_q, nan=0.0, posinf=0.0, neginf=0.0)
    ipd_q[ipd_q < 0.0] = 0.0
    return ipd_q.astype(np.float64, copy=False)


def write_pcap_from_ipds(out_path: Path, ipds_s: np.ndarray, *, start_epoch_s: float = 1_700_000_000.0) -> None:
    """Write a classic libpcap (.pcap) file given an IPD series (seconds)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Classic libpcap global header (little-endian, microsecond resolution)
    # magic = 0xd4c3b2a1
    gh = struct.pack(
        "<IHHIIII",
        0xA1B2C3D4,  # magic (we'll write LE bytes via '<' pack)
        2, 4,        # version 2.4
        0,           # thiszone
        0,           # sigfigs
        65535,       # snaplen
        1,           # network: ethernet
    )

    frame = _build_dummy_eth_ipv4_udp_frame(payload_len=32)
    incl_len = len(frame)
    orig_len = len(frame)

    # Build timestamps
    ipds_s = np.asarray(ipds_s, dtype=np.float64).reshape(-1)
    ts = np.empty(ipds_s.size + 1, dtype=np.float64)
    ts[0] = float(start_epoch_s)
    ts[1:] = ts[0] + np.cumsum(ipds_s)

    with open(out_path, "wb") as f:
        f.write(gh)
        for t in ts:
            sec = int(np.floor(t))
            usec = int(round((t - sec) * 1e6))
            # normalize
            if usec >= 1_000_000:
                sec += 1
                usec -= 1_000_000
            ph = struct.pack("<IIII", sec, usec, incl_len, orig_len)
            f.write(ph)
            f.write(frame)


def generate_demarking_pcap(
    out_path: Path | str,
    *,
    packets: int = 600,
    seed: int = 1337,
    threshold: float = 0.6,
    margin: float = 0.005,
    max_tries: int = 2000,
) -> dict:
    """Generate a DeMarking-test PCAP and (if possible) verify it crosses `threshold`.

    This is safe to call from the NeurInSpectre CLI: it produces a **local**
    classic `.pcap` file with plausible packet bytes and adversarial timing.
    """
    out_path = Path(out_path)
    n = int(packets)
    if n < 10:
        raise ValueError("packets must be >= 10")

    base_seed = int(seed)
    thr = float(threshold)
    m = float(margin)

    chosen_seed = base_seed
    ipds = _timing_spec_ipds(n_packets=n, seed=chosen_seed)

    score = None
    is_evasion = None

    # Double-check (and optionally search for a better seed) if detector is available.
    if DeMarkingDefenseDetector is not None:
        det = DeMarkingDefenseDetector(window_size=50, threshold=thr)
        target = thr + m
        best_score = -1.0
        best_seed = chosen_seed
        best_ipds = _pcap_microsecond_quantize_ipds(ipds)

        for s in range(base_seed, base_seed + max(1, int(max_tries))):
            cand = _timing_spec_ipds(n_packets=n, seed=s)
            cand_q = _pcap_microsecond_quantize_ipds(cand)
            res = det.detect_watermarking_evasion(cand_q)
            sc = float(res.get("evasion_score", 0.0))
            if sc > best_score:
                best_score = sc
                best_seed = s
                best_ipds = cand_q
            if sc >= target:
                chosen_seed = s
                ipds = cand_q  # write quantized IPDs so the file round-trips exactly
                score = sc
                is_evasion = bool(res.get("is_evasion", False))
                break
        else:
            chosen_seed = best_seed
            ipds = best_ipds
            score = float(best_score) if best_score >= 0.0 else None
            is_evasion = bool(score is not None and score >= thr)

    write_pcap_from_ipds(out_path, ipds)

    # Round-trip check (what the CLI will see).
    ipd_roundtrip = _load_pcap_ipds(out_path)
    if DeMarkingDefenseDetector is not None and ipd_roundtrip.size >= 2:
        det = DeMarkingDefenseDetector(window_size=50, threshold=thr)
        res = det.detect_watermarking_evasion(ipd_roundtrip)
        score = float(res.get("evasion_score", 0.0))
        is_evasion = bool(res.get("is_evasion", False))

    return {
        "out_path": str(out_path),
        "seed": int(chosen_seed),
        "threshold": float(thr),
        "margin": float(m),
        "ipd_len": int(ipd_roundtrip.size),
        "demarking_score": (float(score) if score is not None else None),
        "is_evasion": (bool(is_evasion) if is_evasion is not None else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a DeMarking-test PCAP (timing-based).")
    ap.add_argument("--out", default="network_flows.pcap", help="Output .pcap path")
    ap.add_argument("--packets", type=int, default=600, help="Number of packets to write (default: 600)")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed (default: 1337)")
    ap.add_argument("--threshold", type=float, default=0.6, help="Detector threshold you plan to use (default: 0.6)")
    ap.add_argument("--margin", type=float, default=0.005, help="Require score >= threshold+margin (default: 0.005)")
    ap.add_argument("--max-tries", type=int, default=2000, help="Max seeds to try to hit threshold (default: 2000)")
    args = ap.parse_args()

    info = generate_demarking_pcap(
        args.out,
        packets=int(args.packets),
        seed=int(args.seed),
        threshold=float(args.threshold),
        margin=float(args.margin),
        max_tries=int(args.max_tries),
    )

    if info.get("demarking_score") is not None:
        print(
            "[generate_demarking_pcap] "
            f"seed={info['seed']} score={float(info['demarking_score']):.3f} is_evasion={info.get('is_evasion')}"
        )

    print(str(info["out_path"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


