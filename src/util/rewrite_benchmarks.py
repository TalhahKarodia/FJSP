import os 
import inspect
import random
import io

def read_file(source : str, id : int, path : str) -> list[str]:
    if source.startswith('0'):
        target_file = f'Behnke{id}.fjs'
    elif source.startswith('1'):
        target_file = f'BrandimarteMk{id}.fjs'
    elif source.startswith('2a'):
        target_file = f'HurinkSdata{id}.fjs'
    elif source.startswith('2b'):
        target_file = f'HurinkEdata{id}.fjs'
    elif source.startswith('2c'):
        target_file = f'HurinkRdata{id}.fjs'
    elif source.startswith('2d'):
        target_file = f'HurinkVdata{id}.fjs'
    elif source.startswith('3'):
        target_file = f'DPpaulli{id}.fjs'
    elif source.startswith('4'):
        target_file = f'ChambersBarnes{id}.fjs'
    elif source.startswith('5'):
        target_file = f'Kacem{id}.fjs'
    elif source.startswith('6'):
        target_file = f'Fattahi{id}.fjs'

    path += f'\\{source}\\{target_file}'
    file = open(path, 'r')
    return file.readlines()

def write_file(benchmark : list[list[int]], path : str, file_name : str) -> None:
    file = open(path + file_name, 'w')
    # there's probably an easier way to do this
    for line in benchmark:
        output = ''
        for value in line:
            output += str(value) + ' '
        file.write(output + '\n')
    file.close()

def rewrite_benchmark(source : str, id : int, path : str, lower_bound : float = 0.9, upper_bound : float = 1.1, worker_amount : int = 3) -> list[list[int]]:
    file_content : list[str] = read_file(source, id, path)
    
    values = [list(map(float, x.strip('\n').split(' '))) for x in file_content]
    result = []
    result.append([int(values[0][0]), int(values[0][1]), worker_amount])
    #original: 2 2 1 2 1 25 2 30 2 1 1 37 2 1 1 2 32 2 2 1 24 2 33 -> 2 operations, o1 -> 2 workstations, o1 on m1 -> 2 worker -> o1 on m1 with w1 -> 25, o1 on m1 with w2 -> 30, ...
    for line in values[1:]:
        new_line = []
        idx = 0
        new_line.append(int(line[idx]))
        for i in range(int(line[0])): # for each operation
            idx += 1
            workstations = int(line[idx])
            new_line.append(int(line[idx]))
            for j in range(workstations): # for each option
                idx += 1 # workstation
                new_line.append(int(line[idx]))
                # randomly choose amount of workers
                workers = random.randint(1, worker_amount)
                new_line.append(workers)
                options = random.sample(range(1, worker_amount+1), k=workers)
                options.sort()
                idx += 1 # duration
                original_duration = line[idx]
                for k in options: # for each possible worker
                    worker_duration = int(random.uniform(original_duration * lower_bound, original_duration * upper_bound) + 0.5)
                    new_line.append(k)
                    new_line.append(worker_duration)
        result.append(new_line)
    return result

def rewrite_all_from_source(source : str, read_path : str, write_path : str, lower_bound : float = 0.9, upper_bound : float = 1.1, worker_amount : int = 3) -> None:
    full_path = read_path + '/' + source + '/'
    for i in range(len(os.listdir(full_path))):
        result = rewrite_benchmark(source, i+1, read_path, lower_bound, upper_bound, worker_amount)
        write_file(result, write_path + '/', f'{source}_{i+1}_updated.fjs')

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
read_path = currentdir + '/Instances_FJSSP'
write_path = currentdir + '/../worker_instances/'
sources = ['0_BehnkeGeiger', '1_Brandimarte', '2a_Hurink_sdata', '2b_Hurink_edata', '2c_Hurink_rdata', '2d_Hurink_vdata', '3_DPpaulli', '4_ChambersBarnes', '5_Kacem', '6_Fattahi']
"""
    EXAMPLE USAGES:
"""
"""source = 1

# rewrite a specific benchmark
id = 5
result = rewrite_benchmark(source=sources[source], id=id, lower_bound=0.9, upper_bound=1.1, worker_amount=3, path=read_path)
write_file(benchmark=result,path=write_path, file_name=f'{sources[source]}_{id}_updated.fjs')

# rewrite all benchmarks from a specific source
rewrite_all_from_source(sources[source], 0.9, 1.1, 3, read_path, write_path)

# rewrite all benchmarks (with the same parameters)
for i in range(len(sources)):
    source = i
    rewrite_all_from_source(sources[source], 0.9, 1.1, 3, read_path, write_path)"""

def get_available_sources():
    return sources

write_path = currentdir + '/../benchmarks_with_workers/'

def rewrite_all_with_workers(read_path: str, write_path: str):
    for benchmark_source in sources:
        full_path = read_path + benchmark_source + '/'
        for i in range(len(os.listdir(full_path))):
            file_content : list[str] = read_file(benchmark_source, i+1, read_path)
            values = file_content[0].split(' ')
            workstation_amount = int(values[1])
            worker_amount = int(workstation_amount*1.5)
            result = rewrite_benchmark(source=benchmark_source, id=i+1, lower_bound=0.9, upper_bound=1.1, worker_amount=worker_amount, path=read_path)
            write_file(benchmark=result,path=write_path, file_name=f'{benchmark_source}_{i+1}_workers.fjs')

def rewrite_all_from_source_with_workers(source: str, read_path: str, write_path: str):
    full_path = read_path + source + '/'
    for i in range(len(os.listdir(full_path))):
        file_content : list[str] = read_file(source, i+1, read_path)
        values = file_content[0].split(' ')
        workstation_amount = int(values[1])
        worker_amount = int(workstation_amount*1.5)
        result = rewrite_benchmark(source=source, id=i+1, lower_bound=0.9, upper_bound=1.1, worker_amount=worker_amount, path=read_path)
        write_file(benchmark=result,path=write_path, file_name=f'{source}_{i+1}_workers.fjs')

def rewrite_benchmark_with_workers(source : str, id: int, read_path : str, write_path : str) -> None:
    file_content : list[str] = read_file(source, id, read_path)
    values = file_content[0].split(' ')
    workstation_amount = int(values[1])
    worker_amount = int(workstation_amount*1.5)
    result = rewrite_benchmark(source=source, id=id, lower_bound=0.9, upper_bound=1.1, worker_amount=worker_amount, path=read_path)
    write_file(benchmark=result,path=write_path, file_name=f'{source}_{id}_workers.fjs')

"""
    EXAMPLE USAGES:
"""
"""source = 1

# rewrite a specific benchmark with workers
rewrite_benchmark_with_workers(sources[0], 1, read_path, write_path)

# rewrite all benchmarks with workers from a specific source
rewrite_all_from_source_with_workers(sources[0], read_path, write_path)

# rewrite all benchmarks
rewrite_all_with_workers(read_path, write_path)"""