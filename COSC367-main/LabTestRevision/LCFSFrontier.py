from search import *
import math
import heapq

class LocationGraph(ExplicitGraph):
    def __init__(self, nodes, locations, edges, starting_nodes, goal_nodes, estimates=None):
        self.nodes = nodes  
        self.locations = locations
        self.edges = edges
        self._starting_nodes = starting_nodes
        self.goal_nodes = goal_nodes
        self.estimates = estimates

    def outgoing_arcs(self, node):
        """Returns a sequence of Arc objects that go out from the given
        node. The action string is automatically generated.

        """
        arcs = []
        for edge in self.edges:
            if len(edge) == 2:  # if no cost is specified
                tail, head = edge
                x1, y1 = self.locations[tail]
                x2, y2 = self.locations[head]
                cost = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            else:
                tail, head, cost = edge
            if tail == node and Arc(tail, head, str(tail) + '->' + str(head), cost) not in arcs:
                arcs.append(Arc(tail, head, str(tail) + '->' + str(head), cost))
            if head == node and Arc(head, tail, str(head) + '->' + str(tail), cost) not in arcs:
                arcs.append(Arc(head, tail, str(head) + '->' + str(tail), cost))
        return sorted(arcs, key=lambda x : (x[0], x[1]))


class LCFSFrontier(Frontier):
    def __init__(self):
        self.container = []
        self.count = 0
    
    def add(self, path):
        total_cost = 0
        for arc in path:
            total_cost += arc[-1]
        heapq.heappush(self.container, (total_cost, self.count, path))
        self.count += 1
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if len(self.container) > 0:
            return heapq.heappop(self.container)[-1]
        else:
            raise StopIteration   # don't change this one
        
def main():
    print('test 1')
    graph = LocationGraph(nodes=set('ABC'),
                          locations={'A': (0, 0),
                                     'B': (3, 0),
                                     'C': (3, 4)},
                          edges={('A', 'B'), ('B','C'),
                                 ('B', 'A'), ('C', 'A')},
                          starting_nodes=['A'],
                          goal_nodes={'C'})
    
    solution = next(generic_search(graph, LCFSFrontier()))
    print_actions(solution)
    
    print()
    print('test 2')
    graph = LocationGraph(nodes=set('ABC'),
                          locations={'A': (0, 0),
                                     'B': (3, 0),
                                     'C': (3, 4)},
                          edges={('A', 'B'), ('B','C'),
                                 ('B', 'A')},
                          starting_nodes=['A'],
                          goal_nodes={'C'})
    
    solution = next(generic_search(graph, LCFSFrontier()))
    print_actions(solution)
    
    print()
    print('test 3')
    pythagorean_graph = LocationGraph(
        nodes=set("abc"),
        locations={'a': (5, 6),
                   'b': (10,6),
                   'c': (10,18)},
        edges={tuple(s) for s in {'ab', 'ac', 'bc'}},
        starting_nodes=['a'],
        goal_nodes={'c'})
    
    solution = next(generic_search(pythagorean_graph, LCFSFrontier()))
    print_actions(solution)    

if __name__ == '__main__':
    main()