import random

def to_index(job, operation, job_sequence):
    counter = -1
    index = 0
    for i in job_sequence:
        if i == job:
            counter += 1
        if counter == operation:
            return index
        index += 1
    return None

class GreedyFJSSPSolver:

    def __init__(self, durations, job_sequence):
        self.durations = durations
        self.job_sequence = job_sequence
        jobs = set(job_sequence)
        self.counts = [job_sequence.count(job) for job in jobs]

    def determine_next(self, next_operation):
        next_durations = [0] * len(next_operation)
        min_index = float('inf')
        min_duration = float('inf')
        machine = [float('inf')] * len(next_operation)
        min_machine = float('inf')
        for i in range(len(next_operation)):
            if next_operation[i] >= self.counts[i]:
                continue
            index = to_index(i, next_operation[i], self.job_sequence)
            operation_durations = self.durations[index]
            # for FJSSP, use this, for FJSSP-W, extract workers
            next_durations[i] = float('inf')
            for j in range(len(operation_durations)):
                if operation_durations[j] > 0 and operation_durations[j] < next_durations[i]:
                    next_durations[i] = operation_durations[j]
                    machine[i] = j
                elif operation_durations[j] > 0 and operation_durations[j] == next_durations[i] and random.random() < 0.5:
                    next_durations[i] = operation_durations[j]
                    machine[i] = j
        for i in range(len(next_durations)):
            if next_durations[i] > 0:
                if next_durations[i] < min_duration:
                    min_duration = next_durations[i]
                    min_machine = machine[i]
                    min_index = i
                elif next_durations[i] == min_duration and random.random() < 0.5:
                    min_duration = next_durations[i]
                    min_machine = machine[i]
                    min_index = i
        return min_index, min_duration, min_machine
    
    def solve(self):
        next_operation = []
        jobs = set(self.job_sequence)
        for i in jobs:
            next_operation.append(0)

        sequence = [float('inf')] * len(self.durations)
        machines = [float('inf')] * len(self.durations)
        counts = [self.job_sequence.count(job) for job in jobs]
        for i in range(len(self.job_sequence)):
            index, duration, machine = self.determine_next(next_operation)
            machines[to_index(index, next_operation[index], self.job_sequence)] = machine
            next_operation[index] += 1
            sequence[i] = index
        return sequence, machines


class GreedyFJSSPWSolver:

    def __init__(self, durations, job_sequence):
        self.durations = durations
        self.job_sequence = job_sequence
        jobs = set(job_sequence)
        self.counts = [job_sequence.count(job) for job in jobs]

    def determine_next(self, next_operation):
        next_durations = [0] * len(next_operation)
        machine = [float('inf')] * len(next_operation)
        worker = [float('inf')] * len(next_operation)
        min_index = float('inf')
        min_duration = float('inf')

        for i in range(len(next_operation)):
            if next_operation[i] >= self.counts[i]:
                continue
            index = to_index(i, next_operation[i], self.job_sequence)
            operation_durations = self.durations[index]

            next_durations[i] = float('inf')
            for j in range(len(operation_durations)):
                for k in range(len(operation_durations[j])):
                    if operation_durations[j][k] > 0 and operation_durations[j][k] < next_durations[i]:
                        next_durations[i] = operation_durations[j][k]
                        machine[i] = j
                        worker[i] = k
                    elif operation_durations[j][k] > 0 and operation_durations[j][k] == next_durations[i] and random.random() < 0.5:
                        next_durations[i] = operation_durations[j][k]
                        machine[i] = j
                        worker[i] = k
        for i in range(len(next_durations)):
            if next_durations[i] > 0:
                if next_durations[i] < min_duration:
                    min_duration = next_durations[i]
                    min_index = i
                elif next_durations[i] == min_duration and random.random() < 0.5:
                    min_duration = next_durations[i]
                    min_index = i
        return min_index, min_duration, machine[min_index], worker[min_index]
    
    def solve(self):
        next_operation = []
        jobs = set(self.job_sequence)
        for i in jobs:
            next_operation.append(0)

        sequence = [float('inf')] * len(self.durations)
        machines = [float('inf')] * len(self.durations)
        workers = [float('inf')] * len(self.durations)
        counts = [self.job_sequence.count(job) for job in jobs]
        for i in range(len(self.job_sequence)):
            index, duration, machine, worker = self.determine_next(next_operation)
            machines[to_index(index, next_operation[index], self.job_sequence)] = machine
            workers[to_index(index, next_operation[index], self.job_sequence)] = worker
            sequence[i] = index
            next_operation[index] += 1
        return sequence, machines, workers