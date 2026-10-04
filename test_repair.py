"""Check how many assignments repair_chromosome_optimised touches."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.decoder_w import FJSSPWDecoder
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
import glob, random

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)
decoder = FJSSPWDecoder(instance)

ga = FJSSPW_GA(instance, random_seed=0)

# Create a random chromosome
rng = random.Random(0)
chrom = decoder.create_random_chromosome(rng)

# Before repair
before_m = list(chrom['machine_assignment'])
before_w = list(chrom['worker_assignment'])

# Apply repair
repaired = ga.repair_chromosome_optimised(chrom)

# After repair
after_m = repaired['machine_assignment']
after_w = repaired['worker_assignment']

# Count changes
m_changes = sum(1 for a, b in zip(before_m, after_m) if a != b)
w_changes = sum(1 for a, b in zip(before_w, after_w) if a != b)

print(f"Total operations: {len(before_m)}")
print(f"Machine assignment changes: {m_changes} ({m_changes/len(before_m)*100:.1f}%)")
print(f"Worker assignment changes:  {w_changes} ({w_changes/len(before_w)*100:.1f}%)")

# Also: how many chromosomes survive repair unchanged?
print("\n--- Repair survival test ---")
unchanged = 0
for i in range(100):
    chrom = decoder.create_random_chromosome(random.Random(i))
    before = tuple(zip(chrom['machine_assignment'], chrom['worker_assignment']))
    ga.repair_chromosome_optimised(chrom)
    after = tuple(zip(chrom['machine_assignment'], chrom['worker_assignment']))
    if before == after:
        unchanged += 1
print(f"Chromosomes unchanged by repair: {unchanged}/100 ({unchanged}%)")