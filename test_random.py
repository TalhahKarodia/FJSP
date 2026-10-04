"""Verify the decoder responds to chromosome changes."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.decoder_w import FJSSPWDecoder
import random, glob

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)
decoder = FJSSPWDecoder(instance)

print("10 random chromosomes, decoded:")
rng = random.Random(0)
makespans = []
for i in range(10):
    chrom = decoder.create_random_chromosome(rng)
    sched = decoder.decode(chrom)
    makespans.append(sched['makespan'])
    print(f"  Chromosome {i}: makespan = {sched['makespan']}")

print(f"\nMin = {min(makespans)}, Max = {max(makespans)}, "
      f"Range = {max(makespans) - min(makespans)}")

# Also: same chromosome decoded twice should give identical result
print("\nSame chromosome decoded twice:")
chrom = decoder.create_random_chromosome(random.Random(42))
m1 = decoder.decode(chrom)['makespan']
m2 = decoder.decode(chrom)['makespan']
print(f"  First: {m1}, Second: {m2}, Same? {m1 == m2}")