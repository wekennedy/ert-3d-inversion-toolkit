# -*- coding: utf-8 -*-
"""
Pre-field sanity check for an ABEM Spread + Protocol XML pair.

Usage:
    python check_abem_files.py sttestspread.xml sttestprotocol.xml
    python check_abem_files.py sttestspread.xml sttestprotocol.xml --fix

ABEM spread files carry two identifiers per electrode: <Id>, which some
spread generators number per-cable (restarting at 1 on each cable), and
<SwitchAddress>, which is always globally unique across the whole spread.
A protocol numbers its Tx/Rx electrodes 0..N over the whole spread, so if
the instrument resolves those numbers against a non-unique <Id> instead
of against <SwitchAddress>, only part of the spread is reachable - and it
fails silently, simply running the subset of measurements it can resolve.

This script parses both files, reports on spread geometry, checks whether
<Id> and <SwitchAddress> are each unique and whether they agree with each
other, and predicts how many of the protocol's measurements would resolve
under each numbering scheme. Pass --fix to also write a corrected copy of
the spread file with every <Id> overwritten to match its <SwitchAddress>.
"""

import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


# Print a section heading followed by an underline of matching length.
def rule(title):
    print("\n" + title)
    print("-" * len(title))


# Parse a Spread XML file into a list of cables, each holding its electrodes.
# Returns the parsed XML root element (needed later for --fix) and a list of
# dicts of the form {'name': ..., 'electrodes': [...]}, where each electrode
# dict holds id, x, y, z, addr (SwitchAddress) and name.
def read_spread(path):
    root = ET.parse(path).getroot()
    cables = []
    for cab in root.findall('Cable'):
        els = []
        for e in cab.findall('Electrode'):
            els.append(dict(
                id=int(e.findtext('Id')),
                x=float(e.findtext('X')),
                y=float(e.findtext('Y') or 0.0),
                z=float(e.findtext('Z') or 0.0),
                addr=int(e.findtext('SwitchAddress')),
                name=e.findtext('Name')))
        cables.append(dict(name=cab.findtext('Name'), electrodes=els))
    return root, cables


# Parse a Protocol XML file into a list of (tx, rx) electrode-number tuples.
# Each <Measure> element's Tx and Rx text is a whitespace-separated list of
# electrode numbers, which are returned as lists of ints. Returns the parsed
# XML root element alongside the list of measurements.
def read_protocol(path):
    root = ET.parse(path).getroot()
    meas = []
    for m in root.findall('.//Measure'):
        tx = [int(v) for v in m.findtext('Tx').split()]
        rx = [int(v) for v in m.findtext('Rx').split()]
        meas.append((tx, rx))
    return root, meas


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    do_fix = '--fix' in sys.argv
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    spread_path, protocol_path = args[0], args[1]

    sroot, cables = read_spread(spread_path)
    proot, meas = read_protocol(protocol_path)
    els = [e for c in cables for e in c['electrodes']]

    print(f"spread   : {spread_path}")
    print(f"           {sroot.findtext('Name')}")
    print(f"protocol : {protocol_path}")
    print(f"           {proot.findtext('Name')}")

    # Section 1: report cable/electrode counts, Id and SwitchAddress ranges per
    # cable, and per-row (grouped by y,z) electrode spacing and direction.
    rule("1. Spread geometry")
    print(f"   cables     : {len(cables)}")
    print(f"   electrodes : {len(els)}")
    for c in cables:
        ce = c['electrodes']
        rows = defaultdict(list)
        for e in ce:
            rows[(e['y'], e['z'])].append(e)
        print(f"   cable {c['name']}: {len(ce)} electrodes, "
              f"Id {min(e['id'] for e in ce)}-{max(e['id'] for e in ce)}, "
              f"SwitchAddress {min(e['addr'] for e in ce)}-"
              f"{max(e['addr'] for e in ce)}")
        for (y, z), re_ in sorted(rows.items()):
            re_.sort(key=lambda e: e['id'])
            direction = "forward" if re_[-1]['x'] > re_[0]['x'] else "reversed"
            xs = sorted(e['x'] for e in re_)
            steps = {round(b - a, 4) for a, b in zip(xs, xs[1:])}
            print(f"       y={y:g}: {len(re_)} electrodes, "
                  f"x {re_[0]['x']:g} -> {re_[-1]['x']:g} ({direction}), "
                  f"spacing {sorted(steps)}")

    # Section 2: the core check - are <Id> and <SwitchAddress> each unique
    # across the whole spread, and do the two sets of values agree?
    rule("2. Identifier integrity")
    ids = [e['id'] for e in els]
    addrs = [e['addr'] for e in els]
    id_dupes = {k: v for k, v in Counter(ids).items() if v > 1}
    addr_dupes = {k: v for k, v in Counter(addrs).items() if v > 1}

    problems = []
    if id_dupes:
        problems.append('id')
        print(f"   !! <Id> is NOT unique: {len(els)} electrodes but only "
              f"{len(set(ids))} distinct Id values")
        worst = sorted(id_dupes.items())[:5]
        for k, v in worst:
            where = [f"cable {c['name']}" for c in cables
                     for e in c['electrodes'] if e['id'] == k]
            print(f"        Id {k} appears {v}x: {', '.join(where)}")
        print("      Any protocol numbering electrodes across the whole spread")
        print("      can only reach the ones whose Id it can resolve.")
    else:
        print(f"   <Id> unique: {len(set(ids))} values")

    if addr_dupes:
        problems.append('addr')
        print(f"   !! <SwitchAddress> is NOT unique: {addr_dupes}")
    else:
        print(f"   <SwitchAddress> unique: {min(addrs)}-{max(addrs)}")

    if set(ids) == set(addrs):
        print("   <Id> and <SwitchAddress> agree - protocol numbers are "
              "unambiguous whichever the instrument resolves against")
    else:
        problems.append('mismatch')
        print("   !! <Id> and <SwitchAddress> describe different sets.")
        print(f"        Id            : {min(ids)}-{max(ids)}")
        print(f"        SwitchAddress : {min(addrs)}-{max(addrs)}")

    # Section 3: summarize which electrode numbers the protocol references and
    # whether each one can be resolved as a SwitchAddress and/or as an Id.
    rule("3. Protocol")
    used = sorted({n for tx, rx in meas for n in tx + rx})
    print(f"   measurements     : {len(meas)}")
    print(f"   electrode numbers: {min(used)}-{max(used)} ({len(used)} distinct)")

    by_addr = {e['addr']: e for e in els}
    id_count = Counter(ids)
    resolvable_addr = [n for n in used if n in by_addr]
    resolvable_id = [n for n in used if id_count.get(n, 0) >= 1]
    print(f"   resolvable as SwitchAddress : {len(resolvable_addr)}/{len(used)}")
    print(f"   resolvable as Id            : {len(resolvable_id)}/{len(used)}"
          + ("  (ambiguous - duplicated)" if id_dupes else ""))

    # Section 4: for each measurement, check whether every Tx/Rx number in it
    # resolves, first assuming resolution against SwitchAddress, then against
    # Id, and report how many measurements would go through under each rule.
    rule("4. What the instrument will execute")
    ok_addr = sum(1 for tx, rx in meas if all(n in by_addr for n in tx + rx))
    ok_id = sum(1 for tx, rx in meas
                if all(id_count.get(n, 0) >= 1 for n in tx + rx))
    print(f"   if numbers resolve against SwitchAddress : "
          f"{ok_addr}/{len(meas)} measurements")
    print(f"   if numbers resolve against Id            : "
          f"{ok_id}/{len(meas)} measurements")
    if ok_id < len(meas) or ok_addr < len(meas):
        lost = len(meas) - min(ok_addr, ok_id)
        print(f"\n   !! Up to {lost} measurements ({100*lost/len(meas):.0f}%) "
              f"may be silently dropped.")
        unreachable = sorted(set(used) - set(resolvable_id))
        if unreachable:
            print(f"      Numbers with no matching Id: {unreachable[0]}"
                  f"-{unreachable[-1]} ({len(unreachable)} electrodes)")
    else:
        print("   all measurements resolve under either rule")

    # Section 5: verdict, and optionally write a fixed spread file with every
    # <Id> overwritten to match its <SwitchAddress>.
    rule("5. Verdict")
    if not problems:
        print("   No structural problems found.")
        return

    print("   Renumber <Id> to match <SwitchAddress> so every electrode in the")
    print("   spread has one globally unique identifier.")

    if do_fix:
        n = 0
        for cab in sroot.findall('Cable'):
            for e in cab.findall('Electrode'):
                e.find('Id').text = e.findtext('SwitchAddress')
                n += 1
        out = os.path.splitext(spread_path)[0] + '_fixed.xml'
        ET.ElementTree(sroot).write(out, encoding='utf-8',
                                    xml_declaration=True)
        print(f"\n   Wrote {out} with {n} <Id> values renumbered to match "
              f"<SwitchAddress>.")
        print("   Check it against your generator before relying on it.")
    else:
        print("   Re-run with --fix to write a corrected spread alongside "
              "the original.")


if __name__ == '__main__':
    main()
