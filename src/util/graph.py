import copy
import random
import numpy as np
import statistics

def run_n_simulations(s, e, m, w, js, d, uncertainty_parameters, n_simulations, uncertainty_source : str = 'worker', processing_times : bool = False, machine_breakdowns : bool = False, worker_unavailabilites : bool = False):
    results = []
    for i in range(n_simulations):
        g = Graph(s, e, m, w, js)
        g.simulate(d, uncertainty_parameters, processing_times=processing_times, machine_breakdowns=machine_breakdowns, worker_unavailabilities=worker_unavailabilites, uncertainty_source=uncertainty_source)
        results.append(float(max(g.e)))

    robust_makespan = statistics.mean(results)
    robust_makespan_stdev = statistics.stdev(results)
    R = robust_makespan/max(e)
    return results, robust_makespan, robust_makespan_stdev, R


class Graph:

    def __init__(self, s, e, a, w, js, leftshift : bool = False, buffers : list[float] = []):
        self.roots = []
        self.leftshift = leftshift
        self.js = copy.deepcopy(js)
        self.s = copy.deepcopy(s)
        self.e = copy.deepcopy(e)
        self.m = copy.deepcopy(a)
        self.w = copy.deepcopy(w)
        Node.leftshift = leftshift
        if len(buffers) == 0:
            buffers = [0] * len(s)
        self.b = copy.deepcopy(buffers)
        nodes = []
        for i in range(len(s)):
            nodes.append(Node(self.s, self.e, self.m, self.w, self.js, self.b, i))
        for i in range(len(nodes)):
            nodes[i].add_neighbours(self.m, self.w, self.js, nodes, i)
            if len(nodes[i].parents) == 0:
                self.roots.append(nodes[i])
        self.all_nodes = nodes
        self.update()

    def get_vectors(self):
        s = []
        e = []
        m = []
        w = []
        b = [self.all_nodes[i].buffer if len(self.all_nodes[i].children) > 0 else 0.0 for i in range(len(self.all_nodes))]
        for i in range(len(self.all_nodes)):
            s.append(self.all_nodes[i].start)
            e.append(self.all_nodes[i].end)
            m.append(self.all_nodes[i].machine)
            w.append(self.all_nodes[i].worker)
            #b.append(self.all_nodes[i].buffer)
        return s, e, m, w, b

    def add_child(self, current, open_list, closed_list):
        in_open = current in open_list
        in_closed = current in closed_list
        if in_open or in_closed:
            return
        for parent in current.parents:
            if parent not in open_list and parent not in closed_list:
                self.add_child(parent, open_list, closed_list)
        open_list.append(current)

    def real_duration(self, d, wv):
        du = d*(1.0+(wv[2] + random.betavariate(wv[0], wv[1])))
        return du
    
    def update(self):
        open_list = []
        closed_list = []
        open_list.extend(self.roots)
        n_changes = 0
        while len(open_list) > 0:
            current : Node = open_list.pop(0)
            closed_list.append(current)
            for child in current.children:
                self.add_child(child, open_list, closed_list)
            change = current.update_values()
            n_changes += change
        n_ops = len(self.e)
        for i in range(n_ops):
            self.s[i] = self.all_nodes[i].start
            self.b[i] = self.all_nodes[i].buffer
            self.e[i] = self.all_nodes[i].end
        return n_changes

    def simulate_processing_times(self, d, wv, uncertainty_source : str = 'worker'):
        open_list = []
        closed_list = []
        open_list.extend(self.roots)
        changes = 0
        while len(open_list) > 0:
            current = open_list.pop(0)
            closed_list.append(current)
            for parent in current.parents: # should not be necessary
                if parent not in closed_list:
                    print('something went wrong')
            for child in current.children:
                self.add_child(child, open_list, closed_list)
            new_duration = self.real_duration(d[current.operation][current.machine][current.worker], wv[current.worker%len(wv)]) if uncertainty_source == 'worker' else self.real_duration(d[current.operation][current.machine][current.worker], wv[current.machine%len(wv)])
            changes += current.update_time_slot(new_duration)
        return changes + self.update()

    def generate_events(self, up):
        events = []
        makespan = max(self.e)
        for i in range(len(up)):
            currentEvents = []
            t = 0.0
            r = random.expovariate(up[i][0])
            while r > t and r < makespan:
                duration = random.weibullvariate(up[i][1], up[i][2])
                currentEvents.append((r, duration))
                t = r + duration
                r = random.expovariate(up[i][0])
            currentEvents.sort(key=lambda x: x[0])
            events.append(currentEvents)
        return events

    def generate_all_events(self, up):
        events = self.generate_events(up)
        all_events = []
        for i in range(len(events)):
            for j in range(len(events[i])):
                all_events.append((i, events[i][j]))
        all_events.sort(key=lambda x: x[1][0])
        return all_events
    
    def find_affected_operation(self, start, end, machine = -1, worker = -1):
        search = []
        find = 0
        
        if machine == -1:
            # worker unavailability
            search = [node.worker for node in self.all_nodes]
            find = worker
        else:
            search = [node.machine for node in self.all_nodes]
            find = machine
        indices = []
        for i in range(len(search)):
            if search[i] == find:
                if self.s[i] <= end and start <= self.e[i]:
                    indices.append(i)
        if len(indices) == 0:
            return -1
        earliest = indices[0]
        for i in range(1, len(indices)):
            if self.s[indices[i]] < self.s[earliest]:
                earliest = indices[i]
        return earliest

    def simulate_machine_breakdowns(self, d):
        n_machines = len(d[0])
        makespan_original = max(self.e)
        lam = (1.0/n_machines)/makespan_original
        up = []
        for i in range(n_machines):
            machine_lambda = lam * random.uniform(0.9,1.1)
            alpha = random.uniform(makespan_original * 0.1, makespan_original * 0.2)
            up.append([machine_lambda, alpha, 3.602])
        all_events = self.generate_all_events(up)
        operations = []
        for i in range(len(all_events)):
            operation = self.find_affected_operation(all_events[i][1][0], all_events[i][1][1], machine=all_events[i][0])
            if operation != -1:
                operations.append((operation, all_events[i][1]))
        if len(operations) == 0:
            return 0
        operations.sort(key=lambda x: x[1][0])
        for i in range(len(operations)):
            self.all_nodes[operations[i][0]].end = operations[i][1][1]
        return self.update()

    def simulate_worker_unavailabilities(self, d):
        n_workers = len(d[0][0])
        makespan_original = max(self.e)
        lam = ((1.0/n_workers)/makespan_original)/2.0
        up = []
        for i in range(n_workers):
            worker_lambda = lam * random.uniform(0.9,1.1)
            alpha = random.uniform(makespan_original * 0.5, makespan_original * 1.0)
            up.append([worker_lambda, alpha, 3.602])
        all_events = self.generate_all_events(up)
        operations = []
        for i in range(len(all_events)):
            operation = self.find_affected_operation(all_events[i][1][0], all_events[i][1][1], worker=all_events[i][0])
            if operation != -1:
                operations.append((operation, all_events[i][1]))
        if len(operations) == 0:
            return 0
        operations.sort(key=lambda x: x[1][0])
        for i in range(len(operations)):
            self.all_nodes[operations[i][0]].end = operations[i][1][1]
        return self.update()

    def simulate(self, d, wv = None, processing_times : bool = False, machine_breakdowns : bool = False, worker_unavailabilities : bool = False, uncertainty_source : str = 'worker'):
        n_conflicts = 0
        if processing_times:
            n_conflicts += self.simulate_processing_times(d, wv, uncertainty_source)
        if machine_breakdowns:
            n_conflicts += self.simulate_machine_breakdowns(d)
        if worker_unavailabilities:
            n_conflicts += self.simulate_worker_unavailabilities(d)
        return n_conflicts

    def get_predecessors(self, node):
        open_list = [node]
        closed_list = []
        while len(open_list) > 0:
            current = open_list.pop(0)
            closed_list.append(current)
            for parent in current.parents:
                if parent not in open_list and parent not in closed_list:
                    open_list.append(parent)
        return closed_list

    def get_successors(self, node):
        open_list = [node]
        closed_list = []
        while len(open_list) > 0:
            current = open_list.pop(0)
            closed_list.append(current)
            for child in current.children:
                if child not in open_list and child not in closed_list:
                    open_list.append(child)
        return closed_list
    
    def count_parents(self, node):
        return len(self.get_predecessors(node))

    def count_children(self, node):
        return len(self.get_successors(node))

    def makespan(self):
        return max(self.e)

    def plot_data(self, strict : bool = False):
        s = []
        e = []
        m = []
        b = [self.all_nodes[i].buffer if len(self.all_nodes[i].children) > 0 else 0.0 for i in range(len(self.all_nodes))]
        jb = []
        w = []
        l = []
        pre = []
        suc = []
        sequence = []
        c = max([node.end for node in self.all_nodes])
        c_nodes = [node for node in self.all_nodes if node.end == c]
        n_machines = len(list(set([node.machine for node in self.all_nodes])))
        n_workers = len(list(set([node.worker for node in self.all_nodes])))
        crit_nodes = []
        for node in c_nodes:
            crit_nodes.append(node)
            if not strict:
                predecessors = self.get_predecessors(node)
                for predecessor in predecessors:
                    if predecessors not in crit_nodes:
                        crit_nodes.append(predecessor)
            else:
                predecessors = [parent for parent in node.parents]
                while len(predecessors) > 0:
                    most_critical = [predecessors.pop(0)]
                    for parent in predecessors:
                        if node.start-parent.end < node.start-most_critical[0].end:
                            most_critical = [parent]
                        elif node.start-parent.end == node.start-most_critical[0].end:
                            most_critical.append(parent)
                    predecessors = []
                    for parent in most_critical:
                        if parent not in crit_nodes:
                            crit_nodes.append(parent)
                            predecessors.extend(parent.parents)
        is_critnode = [True if node in crit_nodes else False for node in self.all_nodes]
        for i in range(len(self.all_nodes)):
            s.append(self.all_nodes[i].start)
            e.append(self.all_nodes[i].end)
            m.append(self.all_nodes[i].machine)
            #b.append(self.all_nodes[i].buffer)
            jb.append(self.all_nodes[i].job)
            w.append(self.all_nodes[i].worker)
            pre.append(self.count_parents(self.all_nodes[i]))
            suc.append(self.count_children(self.all_nodes[i]))
            
            l.append([])
            for child in self.all_nodes[i].children:
                l[-1].append([e[-1],child.start, m[-1], child.machine])
            sequence.append(self.all_nodes[i].operation)
        js = sorted(sequence, key=lambda x: s[x])
        on_machine = []
        for i in range(n_machines): # n machines
            machine = []
            for j in range(len(js)):
                if m[j] == i:
                    machine.append(j)
            machine.sort(key=lambda x: s[x])
            on_machine.append(machine)
        same_worker = []
        for i in range(n_workers): # n workers
            worker = []
            for j in range(len(js)):
                if w[j] == i:
                    worker.append(j)
            worker.sort(key=lambda x: s[x])
            same_worker.append(worker)
        job_sequence = []
        for i in range(len(set(js))):
            job = [0] * len(js)
            for j in range(js.index(i), js.index(i)+js.count(i)):
                operation = j-js.index(i)
                n_operations = js.count(i)
                job[j] = n_operations-operation-1
            job_sequence.append(job)
        return s, e, m, w, b, jb, l, pre, suc, job_sequence, on_machine, same_worker, is_critnode

class Node:

    leftshift = False

    def __init__(self, s, e, m, w, js, b, i):
        self.start = s[i]
        self.end = e[i]
        self.job = js[i]
        self.machine = m[i]
        self.worker = w[i]
        self.parents = []
        self.children = []
        self.buffer = b[i]#*(self.end-self.start)
        
        self.operation = i

    def add_neighbours(self, m, w, js, nodes, i):
        if i > 0 and js[i-1] == js[i]:
            self.parents.append(nodes[i-1])
        if i+1 < len(js) and js[i+1] == js[i]:
            self.children.append(nodes[i+1])
        on_machine = [j for j in range(len(m)) if m[j] == m[i]]
        on_machine.sort(key=lambda x: nodes[x].start)
        mi = on_machine.index(i)
        if mi > 0:
            idx = mi-1
            while idx >= 0 and nodes[on_machine[idx]].start == nodes[on_machine[mi]].start:
                idx-=1
            if nodes[on_machine[idx]].start != nodes[on_machine[mi]].start:
                self.parents.append(nodes[on_machine[idx]])
        if mi+1 < len(on_machine):
            self.children.append(nodes[on_machine[mi+1]])
        same_worker = [j for j in range(len(w)) if w[j] == w[i]]
        same_worker.sort(key=lambda x: nodes[x].start)
        wi = same_worker.index(i)
        if wi > 0:
            idx = wi-1
            while idx >= 0 and nodes[same_worker[idx]].start == nodes[same_worker[wi]].start:
                idx-=1
            if nodes[same_worker[idx]].start != nodes[same_worker[wi]].start:

                self.parents.append(nodes[same_worker[idx]])
        if wi+1 < len(same_worker):
            self.children.append(nodes[same_worker[wi+1]])

    def update_values(self):
        s = self.start
        e = self.end
        d = e-s
        change = 0
        parent_end_times = [parent.end + ((parent.end-parent.start)*parent.buffer) for parent in self.parents]
        parent_end_times.append(s)
        earliest_start = max(parent_end_times)# if len(parent_end_times) > 0 else 0
        if earliest_start > s:# or (Node.leftshift and earliest_start < s):
            s = earliest_start
            e = s + d
            change = 1
        self.start = s
        self.end = e
        return change

    def update_time_slot(self, du):
        s = self.start
        #d = e - s # determine real duration here
        d = du#[self.operation][self.machine][self.worker]
        e = s + d#self.end
        change = 0
        parent_end_times = [parent.end + ((parent.end-parent.start)*parent.buffer) for parent in self.parents]
        parent_end_times.append(s)
        earliest_start = max(parent_end_times)# if len(parent_end_times) > 0 else 0
        if earliest_start > s or (Node.leftshift and earliest_start < s):
            s = earliest_start
            e = s + d
            change = 1
        self.start = s
        #self.buffer = max(0, self.end+self.buffer - e)
        self.end = e
        return change