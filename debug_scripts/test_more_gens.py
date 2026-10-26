"""Check if more generations close the gap."""
import sys, io, contextlib, glob, os, time
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from baselines.standard_ga import StandardGA

paths = sorted(glob.glob("../FJSSP-W-Competition/instances/fjssp-w/*BehnkeGeiger_42*"))
inst = parse_competition_instance(paths[0])
print(f"Instance: {os.path.basename(paths[0])}")

print("\n=== Proposed FPC-GA with varying pop/gen (balance=0) ===")
for pop, gen in [(10, 10), (30, 30), (30, 50), (50, 100)]:
    t0 = time.time()
    ga = FJSSPW_GA(inst, population_size=pop, generations=gen, random_seed=0,
                   fitness_mode='deterministic', balance_weight=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        r = ga.evolve()
    print(f"  pop={pop:>3}, gen={gen:>3}: makespan={r['best_makespan']:.0f}, "
          f"fev={r['fev_count']}, {time.time()-t0:.2f}s")

print("\n=== Standard GA (for reference) ===")
for pop, gen in [(10, 10), (30, 50)]:
    sga = StandardGA(inst, population_size=pop, generations=gen, random_seed=0)
    r = sga.evolve()
    print(f"  pop={pop:>3}, gen={gen:>3}: makespan={r['best_makespan']:.0f}")
