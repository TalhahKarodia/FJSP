import numpy as np

class Encoding:
    def __init__(self, durations: list, job_sequence: list) -> None:
        self.__durations = durations
        self.__job_sequence = job_sequence
        self.__n_jobs = 1
        for i in range(1, len(self.__job_sequence)):
            if self.__job_sequence[i] != self.__job_sequence[i - 1]: 
                self.__n_jobs += 1

    def job_sequence(self):
        return self.__job_sequence
    
    def n_operations(self):
        return self.__durations.shape[0]
    
    def n_machines(self):
        return self.__durations.shape[1]
    
    def n_jobs(self):
        return self.__n_jobs
    
    def durations(self):
        return self.__durations
    
    def get_machines_for_operation(self, operation_index: int):
        machines = []
        for i in range(0, self.__durations.shape[1]): 
            if self.__durations[operation_index, i] > 0:
                machines.append(i)
        return machines
    
    def get_machines_for_all_operations(self):
        machines = []
        for i in range (0, self.__durations.shape[0]):
            machines.append([])
            for j in range(0, self.__durations.shape[1]):
                if self.__durations[i, j] > 0: 
                    machines[i].append(j)
        return machines
    
    def copy(self):
        return Encoding(self.__durations, self.__job_sequence)
    
    def deep_copy(self):
        duration_copy = np.zeros((self.__durations.shape[0], self.__durations.shape[1]), dtype=int)
        for i in range(0, self.__durations.shape[0]):
            for j in range(0, self.__durations.shape[1]):
                duration_copy[i, j] = self.__durations[i, j]
        
        job_sequence_copy = [None] * len(self.__job_sequence)
        for i in range(0, len(self.__job_sequence)):
            job_sequence_copy[i] = self.__job_sequence[i]

        return Encoding(duration_copy, job_sequence_copy)

class WorkerEncoding:
    def __init__(self, durations: list, job_sequence: list) -> None:
        self.__durations = durations
        self.__job_sequence = job_sequence
        self.__n_jobs = 1
        for i in range(1, len(self.__job_sequence)):
            if self.__job_sequence[i] != self.__job_sequence[i - 1]: 
                self.__n_jobs += 1

    def job_sequence(self):
        return self.__job_sequence
    
    def n_operations(self):
        return self.__durations.shape[0]
    
    def n_machines(self):
        return self.__durations.shape[1]
    
    def n_jobs(self):
        return self.__n_jobs
    
    def durations(self):
        return self.__durations

    def get_workers_for_operation(self, operation_index: int):
        workers = []
        for i in range(0, self.__durations.shape[1]): 
            for j in range(0, self.__durations.shape[2]):
                if self.__durations[operation_index, i, j] > 0:
                    workers.append(j)

        return workers
    
    def get_all_machines_for_all_operations(self):
        operations = []
        for i in range (0, self.__durations.shape[0]):
            machines = []
            for j in range(0, self.__durations.shape[1]):
                for k in range(0, self.__durations.shape[2]):
                    if self.__durations[i, j, k] > 0: 
                        machines.append(j)
                        break
                    
            operations.append(machines)

        return operations
    
    def get_workers_for_operation_on_machine(self, operation: int, machine: int):
        workers = []
        for j in range(0, self.__durations.shape[2]):
            if self.__durations[operation, machine, j] > 0:
                workers.append(j)

        return workers
    
    def is_possible(self, operation: int, machine: int, worker: int):
        return self.__durations[operation, machine, worker] > 0

    def copy(self):
        return WorkerEncoding(self.__durations, self.__job_sequence)
    
    def deep_copy(self):
        duration_copy = np.zeros((self.__durations.shape[0], self.__durations.shape[1], self.__durations.shape[2]), dtype=int)
        for i in range(0, self.__durations.shape[0]):
            for j in range(0, self.__durations.shape[1]):
                for k in range(0, self.__durations.shape[2]):
                    duration_copy[i, j, k] = self.__durations[i, j, k]
        
        job_sequence_copy = [None] * len(self.__job_sequence)
        for i in range(0, len(self.__job_sequence)):
            job_sequence_copy[i] = self.__job_sequence[i]

        return WorkerEncoding(duration_copy, job_sequence_copy)
