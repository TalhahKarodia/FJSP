"""Inspect job_sequence format expected by competition's Graph."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
import glob
from collections import Counter

# Find the Fattahi20 instance
path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)

# 1. Inspect job_sequence as returned by our parser
js = instance['job_sequence']
print("=" * 60)
print("instance['job_sequence']")
print("=" * 60)
print(f"Type: {type(js)}")
print(f"Length: {len(js) if hasattr(js, '__len__') else 'N/A'}")

if isinstance(js, list):
    print(f"First 20 elements: {js[:20]}")
    print(f"Last 10 elements:  {js[-10:]}")
    
    if all(isinstance(x, int) for x in js):
        print(f"Unique values: {sorted(set(js))}")
        counts = Counter(js)
        print(f"Occurrences per job: {dict(counts)}")
    elif all(isinstance(x, list) for x in js):
        print(f"First element type: {type(js[0])}")
        print(f"First element: {js[0]}")
        print(f"Lengths of first 5 sublists: {[len(x) for x in js[:5]]}")
else:
    print(f"Contents (first 20 chars if str): {str(js)[:200]}")

# 2. Inspect encoding object
enc = instance['encoding']
print("\n" + "=" * 60)
print("encoding object")
print("=" * 60)
print(f"Type: {type(enc)}")
methods = [m for m in dir(enc) if not m.startswith('_')]
print(f"Methods: {methods}")

# 3. Call encoding.job_sequence()
print("\n" + "=" * 60)
print("encoding.job_sequence()")
print("=" * 60)
try:
    enc_js = enc.job_sequence()
    print(f"Type: {type(enc_js)}")
    print(f"Length: {len(enc_js)}")
    print(f"First 20: {enc_js[:20]}")
    print(f"Last 10:  {enc_js[-10:]}")
    print(f"Same as instance['job_sequence']? {enc_js == js}")
except Exception as e:
    print(f"Error: {e}")

# 4. Inspect durations
print("\n" + "=" * 60)
print("instance['durations']")
print("=" * 60)
d = instance['durations']
print(f"Outer type: {type(d)}")
print(f"Outer length (n_ops): {len(d)}")
if d:
    print(f"d[0] type: {type(d[0])}")
    print(f"d[0] length (n_machines): {len(d[0])}")
    if d[0]:
        print(f"d[0][0] type: {type(d[0][0])}")
        print(f"d[0][0] length (n_workers): {len(d[0][0])}")
        print(f"d[0][0] contents: {d[0][0]}")

# 5. Inspect encoding internals
print("\n" + "=" * 60)
print("Encoding internals (attributes)")
print("=" * 60)
for attr in dir(enc):
    if attr.startswith('_'):
        continue
    try:
        val = getattr(enc, attr)
        if callable(val):
            continue
        # Truncate output
        s = str(val)
        if len(s) > 200:
            s = s[:200] + "..."
        print(f"  {attr}: {s}")
    except Exception as e:
        print(f"  {attr}: ERROR {e}")

# 6. Try to inspect what Graph expects
print("\n" + "=" * 60)
print("Graph expectations from competition example")
print("=" * 60)
print("From competition docs:")
print("  instance = {'s': [...], 'm': [...], 'w': [...], 'd': encoding.durations(), 'js': encoding.job_sequence()}")
print("  Graph(s, e, m, w, js) where e = s[i] + d[i][m[i]][w[i]]")
print(f"\nOur instance:")
print(f"  n_jobs={instance['n_jobs']}, n_machines={instance['n_machines']}, "
      f"n_workers={instance['n_workers']}, n_ops={instance['n_operations']}")