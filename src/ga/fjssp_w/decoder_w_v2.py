"""
Worker-Aware Decoder for FJSSP-W (optimised)

Decodes a chromosome (machine assignment, worker assignment, operation
sequence) into a feasible schedule using Topological List Decoding.

Behaviour is identical to the previous version; speed-ups:
- Heap-based ready list: O(n log J) per decode instead of O(n * J) scans.
- Priorities in a list instead of a dict, no lambda in the hot loop.
- Eligible workers / feasible pairs / greedy-sorted pairs precomputed once.
- Feasibility check memoised (encoding.is_possible called once per triple).
"""

import random
from heapq import heapify, heappop, heappush
from typing import List, Dict


class FJSSPWDecoder_V2:
    """
    Chromosome structure:
    1. machine_assignment: [m0, m1, ..., m_(n_ops-1)]
    2. worker_assignment:  [w0, w1, ..., w_(n_ops-1)]
    3. operation_sequence: permutation of [0..n_ops-1]
    """

    def __init__(self, instance_data):
        self.n_jobs = instance_data['n_jobs']
        self.n_machines = instance_data['n_machines']
        self.n_workers = instance_data['n_workers']
        self.n_operations = instance_data['n_operations']
        # numpy 3-D arrays are slow to index element by element; plain nested
        # lists are several times faster in the decode loop (values are identical).
        _d = instance_data['durations']
        self.durations = _d.tolist() if hasattr(_d, 'tolist') else _d
        self.job_sequence = instance_data['job_sequence']
        self.encoding = instance_data['encoding']

        self.op_to_job = {}
        self.job_op_indices = {}
        for op_idx, job_id in enumerate(self.job_sequence):
            self.op_to_job[op_idx] = job_id
            self.job_op_indices.setdefault(job_id, []).append(op_idx)

        self.op_position_in_job = {}
        for job_id, ops in self.job_op_indices.items():
            for pos, op_idx in enumerate(ops):
                self.op_position_in_job[op_idx] = pos

        # Intra-job successor links and the first op of every job
        n = self.n_operations
        self.next_op = [-1] * n
        self.first_ops = []
        for job_id in range(self.n_jobs):
            ops = self.job_op_indices.get(job_id, [])
            if ops:
                self.first_ops.append(ops[0])
                for a, b in zip(ops, ops[1:]):
                    self.next_op[a] = b

        # Precomputed eligibility tables
        nm = self.n_machines
        enc = self.encoding
        self._eligible = [
            [[int(w) for w in enc.get_workers_for_operation_on_machine(op, m)] for m in range(nm)]
            for op in range(n)
        ]
        self.feasible_pairs = [
            [(m, w) for m in range(nm) for w in self._eligible[op][m]]
            for op in range(n)
        ]
        self._greedy_sorted = [None] * n
        self._ok = set()  # verified-feasible (op, m, w) encoded as ints

    # ------------------------------------------------------------------
    def decode(self, chromosome: Dict) -> Dict:
        ma = chromosome['machine_assignment']
        wa = chromosome['worker_assignment']
        seq = chromosome['operation_sequence']

        n = self.n_operations
        assert len(ma) == n, f"Machine assignment length {len(ma)} != {n}"
        assert len(wa) == n, f"Worker assignment length {len(wa)} != {n}"
        assert len(seq) == n, f"Operation sequence length {len(seq)} != {n}"

        prio = [0] * n
        for rank, op in enumerate(seq):
            prio[op] = rank

        heap = [(prio[op], op) for op in self.first_ops]
        heapify(heap)

        start = [0] * n
        end = [0] * n
        ready = [0] * n
        nm = self.n_machines
        nw = self.n_workers
        mend = [0] * nm
        wend = [0] * nw
        durations = self.durations
        next_op = self.next_op
        ok = self._ok
        makespan = 0

        while heap:
            op = heappop(heap)[1]
            m = ma[op]
            w = wa[op]

            key = (op * nm + m) * nw + w
            if key not in ok:
                if not self.encoding.is_possible(op, m, w):
                    raise ValueError(
                        f"Operation {op}: Machine {m} + Worker {w} is infeasible!"
                    )
                ok.add(key)

            s = mend[m]
            t = wend[w]
            if t > s:
                s = t
            t = ready[op]
            if t > s:
                s = t

            e = s + durations[op][m][w]
            start[op] = s
            end[op] = e
            mend[m] = e
            wend[w] = e
            if e > makespan:
                makespan = e

            nxt = next_op[op]
            if nxt >= 0:
                ready[nxt] = e
                heappush(heap, (prio[nxt], nxt))

        return {
            'start_times': start,
            'end_times': end,
            'machine_assignment': ma,
            'worker_assignment': wa,
            'makespan': makespan,
        }


    # ------------------------------------------------------------------
    def decode_makespan(self, chromosome: Dict):
        """Makespan only. Same schedule logic as decode() but without building
        start/end arrays or the result dict; used on the evaluation hot path."""
        ma = chromosome['machine_assignment']
        wa = chromosome['worker_assignment']
        seq = chromosome['operation_sequence']
        n = self.n_operations
        if len(ma) != n or len(wa) != n or len(seq) != n:
            raise AssertionError("Chromosome length mismatch")

        prio = [0] * n
        for rank, op in enumerate(seq):
            prio[op] = rank

        # heap of plain int ranks; seq[rank] recovers the operation
        heap = [prio[op] for op in self.first_ops]
        heapify(heap)

        ready = [0] * n
        mend = [0] * self.n_machines
        wend = [0] * self.n_workers
        durations = self.durations
        next_op = self.next_op
        ok = self._ok
        nm = self.n_machines
        nw = self.n_workers
        makespan = 0

        while heap:
            op = seq[heappop(heap)]
            m = ma[op]
            w = wa[op]

            key = (op * nm + m) * nw + w
            if key not in ok:
                if not self.encoding.is_possible(op, m, w):
                    raise ValueError(
                        f"Operation {op}: Machine {m} + Worker {w} is infeasible!"
                    )
                ok.add(key)

            s = mend[m]
            t = wend[w]
            if t > s:
                s = t
            t = ready[op]
            if t > s:
                s = t

            e = s + durations[op][m][w]
            mend[m] = e
            wend[w] = e
            if e > makespan:
                makespan = e

            nxt = next_op[op]
            if nxt >= 0:
                ready[nxt] = e
                heappush(heap, prio[nxt])

        return makespan

    # ------------------------------------------------------------------
    def get_processing_time(self, op_idx: int, machine: int, worker: int) -> float:
        return self.durations[op_idx][machine][worker]

    def is_feasible_assignment(self, op_idx: int, machine: int, worker: int) -> bool:
        return self.encoding.is_possible(op_idx, machine, worker)

    def get_eligible_workers(self, op_idx: int, machine: int) -> List[int]:
        return self._eligible[op_idx][machine]

    def get_eligible_machines(self, op_idx: int) -> List[int]:
        el = self._eligible[op_idx]
        return [m for m in range(self.n_machines) if el[m]]

    # ------------------------------------------------------------------
    def create_random_chromosome(self, rng: random.Random = None) -> Dict:
        if rng is None:
            rng = random.Random()

        n_ops = self.n_operations
        machine_assignment = []
        worker_assignment = []
        for op_idx in range(n_ops):
            pairs = self.feasible_pairs[op_idx]
            if not pairs:
                raise ValueError(f"No feasible pair for operation {op_idx}")
            m, w = rng.choice(pairs)
            machine_assignment.append(m)
            worker_assignment.append(w)

        operation_sequence = list(range(n_ops))
        rng.shuffle(operation_sequence)
        return {
            'machine_assignment': machine_assignment,
            'worker_assignment': worker_assignment,
            'operation_sequence': operation_sequence,
        }

    def _sorted_feasible(self, op_idx: int):
        lst = self._greedy_sorted[op_idx]
        if lst is None:
            d = self.durations[op_idx]
            lst = sorted(
                (d[m][w], m, w)
                for (m, w) in self.feasible_pairs[op_idx]
                if d[m][w] > 0
            )
            self._greedy_sorted[op_idx] = lst
        return lst

    def create_greedy_chromosome(self, rng: random.Random = None,
                                 greedy_prob: float = 0.7) -> Dict:
        """With prob `greedy_prob` pick the cheapest pair, else a random one
        from the cheapest 10%."""
        if rng is None:
            rng = random.Random()

        n_ops = self.n_operations
        machine_assignment = []
        worker_assignment = []
        for op_idx in range(n_ops):
            feasible = self._sorted_feasible(op_idx)
            if not feasible:
                raise ValueError(f"No feasible pair for operation {op_idx}")

            if rng.random() < greedy_prob:
                _, m, w = feasible[0]
            else:
                top_k = max(1, len(feasible) // 10)
                _, m, w = rng.choice(feasible[:top_k])

            machine_assignment.append(m)
            worker_assignment.append(w)

        operation_sequence = list(range(n_ops))
        rng.shuffle(operation_sequence)
        return {
            'machine_assignment': machine_assignment,
            'worker_assignment': worker_assignment,
            'operation_sequence': operation_sequence,
        }