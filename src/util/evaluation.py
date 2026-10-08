import numpy as np

def makespan(start_times : list[int], machine_assignments : list[int], worker_assignments : list[int], durations : list[list[list[int]]]) -> float:
    # NOTE: assume first operation to start at t = 0
    return np.max([start_times[i] + durations[i][machine_assignments[i]][worker_assignments[i]] for i in range(len(start_times))])

def workload_balance(machine_assignments : list[int], worker_assignments : list[int], durations : list[list[list[int]]]) -> float:
    n_workers = max(worker_assignments)
    working_time = [0] * (n_workers+1)
    for i in range(len(worker_assignments)):
        working_time[worker_assignments[i]] += durations[i][machine_assignments[i]][worker_assignments[i]]
    mean_working_time = np.mean(working_time)
    result = 0.0
    for i in range(len(working_time)):
        result += np.pow((mean_working_time - working_time), 2)
    return result

def makespan_fjssp(start_times : list[int], machine_assignments : list[int], durations : list[list[list[int]]]) -> float:
    return np.max([start_times[i] + durations[i][machine_assignments[i]] for i in range(len(start_times))])

def translate_fjssp(sequence : list[int], machines : list[int], durations : list[list[list[int]]]) -> tuple[list[int], list[int]]:
    def get_start_index(job : int, job_sequence : list[int]) -> int:
        for i in range(len(job_sequence)):
            if job_sequence[i] == job:
                return i
        return None
    job_sequence = sorted(sequence)
    jobs = sorted(list(set(sequence)))

    n_machines = len(durations[0])
    n_operations = len(sequence)
    next_operations = [0] * len(jobs)
    end_on_workstations = [0] * n_machines
    end_times = [0] * n_operations
    start_times = [0] * n_operations
    start_indices = [get_start_index(job, job_sequence) for job in jobs]
    for i in range(len(sequence)):
        job = sequence[i]
        operation = next_operations[job]
        next_operations[job] += 1
        start_index = start_indices[job] + operation
        machine = machines[start_index]
        duration = durations[start_index]
        offset = 0
        if operation > 0:
            offset = max(0, end_times[start_index-1] - end_on_workstations[machine])
        end_times[start_index] = end_on_workstations[machine]+duration+offset
        start_times[start_index] = end_on_workstations[machine] + offset
        end_on_workstations[machine] = end_times[start_index]
    return start_times, machines

def translate(sequence : list[int], machines : list[int], workers : list[int], durations : list[list[list[int]]]) -> tuple[list[int], list[int], list[int]]:
    
    class TimeSlot:
        def __init__(self) -> None:
            self.start = 0
            self.end = 0

        def __init__(self, start : int, end : int) -> None:
            self.start = start
            self.end = end

        def overlaps(self, other) -> bool:
            return self.contains(other.start) or self.contains(other.end) or other.contains(self.start) or other.contains(self.end)

        def contains(self, time : int) -> bool:
            return time >= self.start and time <= self.end
    
    def get_start_index(job : int, job_sequence : list[int]) -> int:
        for i in range(len(job_sequence)):
            if job_sequence[i] == job:
                return i
        return None

    def earliest_fit(times : list[TimeSlot], slot : TimeSlot) -> TimeSlot:
        for i in range(1, len(times)):
            if times[i-1].end <= slot.start and times[i].start >= slot.end:
                return times[i-1]
        return times[-1]

    job_sequence = sorted(sequence)
    jobs = set(job_sequence)
    start_indices = [get_start_index(job, job_sequence) for job in sorted(list(jobs))]
    n_operations = len(durations)
    n_machines = len(durations[0])
    n_workers = len(durations[0][0])
    next_operation : list[int] = [0] * len(jobs)
    end_on_machines : list[TimeSlot] = [[TimeSlot(0, 0)] for _ in range(n_machines)]
    end_of_workers : list[TimeSlot] = [[TimeSlot(0, 0)] for _ in range(n_workers)]
    start_times : list[int] = [0] * n_operations
    end_times : list[int] = [0] * n_operations
    for i in range(n_operations):
        job = sequence[i]
        operation = next_operation[job]
        next_operation[job] += 1
        start_index = start_indices[job] + operation
        machine = machines[start_index]
        worker = workers[start_index]
        duration = durations[start_index][machine][worker]
        if duration == 0:
            raise Exception("Invalid solution")
        offset = 0
        if operation > 0:
            if end_times[start_index - 1] > offset:
                offset = end_times[start_index - 1]
        if len(end_on_machines[machine]) > 0 and end_on_machines[machine][-1].end >= offset:
            offset = end_on_machines[machine][-1].end
        if len(end_of_workers[worker]) > 0:
            worker_earliest : TimeSlot = earliest_fit(end_of_workers[worker], TimeSlot(offset, offset + duration))
            if worker_earliest.end >= offset:
                offset = worker_earliest.end
        start_times[start_index] = offset
        end_times[start_index] = offset + duration
        end_on_machines[machine].append(TimeSlot(offset, offset + duration))
        end_of_workers[worker].append(TimeSlot(offset, offset + duration))
        #end_on_machines[machine].sort(key=lambda x: x.start) # should be sorted anyway
        #end_of_workers[worker].sort(key=lambda x: x.start)
    return start_times, machines, workers

def minizinc_score(data : dict[str, dict[str, tuple[float, float]]], ignoreCompletionTime : bool = False) -> dict[str, float]:
    scores = dict()
    for solver in data:
        scores[solver] = 0.0
        for instance in data[solver]:
            if ignoreCompletionTime:
                fitness = data[solver][instance]#
                time = 0
            else:
                time = data[solver][instance][0]
                fitness = data[solver][instance][1]
            for other in data:
                if other != solver:
                    if instance in data[other]:
                        if ignoreCompletionTime:
                            other_time = 0
                            other_fitness = data[other][instance]
                        else:
                            other_time = data[other][instance][0]
                            other_fitness = data[other][instance][1]
                        if fitness < other_fitness:
                            scores[solver] += 1.0
                        if fitness == other_fitness:
                            if ignoreCompletionTime:
                                scores[solver] += 0.5
                            else:
                                scores[solver] += other_time / (other_time + time)
                    else:
                        scores[solver] += 1.0
    return scores