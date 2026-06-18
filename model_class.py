""" PERSONAL NOTES AND COMMENTS
    This program is a simple implementation of the energy price aware network mapping problem for Infrastructure Providers (InP) to reduce their energy costs. The first implementation will take only into account a fraction of the constraints in my draft, and with a not-yet proper modeling of the physical and logical graphs, but it will be a good starting point to test the model and then we can start to add more constraints and a better modeling of the graphs.

    I recall having a program that generates random graphs, it will be useful to test the model with different topologies and different parameters but that's not the purpose of this first draft

    

    Created in June 16th, 2026 by Enzo Henry
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import json

def json_parser(model_name: str, file_name = "test_models.json") -> dict:
    """
        json_parser returns a dict containing every paramters for the model (physGraph, sfc, availability, requirements etc)
        Arguments:  model_name      "model1" for example
                    file_name       not needed here
    """
    with open(file_name, 'r') as file:
        test_models = json.load(file)["test_models"]
    #print(test_models["test_models"])
    for model in test_models:
        if model["model_name"] == model_name:
            return model["model_parameters"]
    raise ValueError(f"Model '{model_name}' does not exist")



# Once our program runs correctly, we can start to implement the classes that will represent our model. We will create a class for the physical servers, a class for the VNFs, and a class for the physical links. We will also create a class for the logical links that will connect the VNFs. Finally, we will create a class for the network mapping that will contain all of the other classes. 
class Server:
    def __init__(self, name, memory=0, CPU=0):
        self.name = name
        self.memory = memory
        self.CPU = CPU
        self.vnfs = []  # List of VNFs hosted on the server
class VNF:
    def __init__(self, name, memory=0, CPU=0):
        self.name = name
        self.memory = memory
        self.CPU = CPU
class PhysLink:
    def __init__(self, server1: Server, server2: Server, bandwidth=0):
        self.server1 = server1
        self.server2 = server2
        self.bandwidth = bandwidth
class LogicalLink:
    def __init__(self, vnf1: VNF, vnf2: VNF, bandwidth=0):
        self.vnf1 = vnf1
        self.vnf2 = vnf2
        self.bandwidth = bandwidth


class NetworkMapping:
    def __init__(self):
        self.gpmodel = gp.Model("mip1")
        self.physGraph = {"i1": ["i2", "i3"], "i2": ["i1", "i4"], "i3": ["i1", "i4"], "i4": ["i2", "i3"]}
        self.sfc = ["v1", "v2", "v3"]
        self.logical_links = [(self.sfc[j], self.sfc[j+1]) for j in range(len(self.sfc)-1)]
        self.physical_links = self.__generate_edges()
        self.physical_link_index = {tuple(edge): idx for idx, edge in enumerate(self.physical_links)}

        # model parameters : later they will be in their own class, but now we just want to test the model with some dummy values
        # ah and only computing for the moment. Memory is analogous and I still have to figure out a good way to model bandwidth and edges
        self.computing_availability = {"i1": 5, "i2": 5, "i3": 5, "i4": 5}
        self.memory_availability    = {"i1": 5, "i2": 5, "i3": 5, "i4": 5}  # not implemented in the following yet
        self.energy_price           = {"i1": 1, "i2": 2, "i3": 3, "i4": 1}
        self.computing_requirements = {"v1": 2, "v2": 2, "v3": 2}

    def __init__(self, model: dict):
        self.gpmodel = gp.Model("mip1")
        self.physGraph = model["physGraph"]
        self.sfc = model["sfc"]
        self.logical_links = [(self.sfc[j], self.sfc[j+1]) for j in range(len(self.sfc)-1)]
        self.physical_links = self.__generate_edges()       # list of 2-list representing an edge 
        self.physical_link_index = {tuple(edge): idx for idx, edge in enumerate(self.physical_links)}

        self.computing_availability = model["computing_availability"]
        self.memory_availability    = model["memory_availability"]
        self.bandwidth_availability = [model["bandwidth_availability_dict"][vertex][neighbor] for vertex, neighbor in self.physical_links]

        self.energy_price           = model["energy_price"]
        self.computing_requirements = model["computing_requirements"]
        self.memory_requirements    = model["memory_requirements"]
        self.bandwidth_requirement  = model["bandwidth_requirements"]       # meant to be a simple list

    def vertices_P(self):
        """ returns the vertices of the graph """
        return list(self.physGraph.keys())
    
    def edges_P(self):
        """ returns the edges of the graph """
        # return self.__generate_edges()
        return self.physical_links
    
    def __generate_edges(self):
        """ generates the edges of the graph obtained by BFS from i1 to last, as a list of 2-lists, each 2-list representing an edge """
        edges = []
        for vertex in self.physGraph:
            for neighbour in self.physGraph[vertex]:
                if (neighbour, vertex) not in edges:
                    edges.append([vertex, neighbour])
        return edges

    def generate_mapping_variables(self):
        """ generates the mapping variables for the VNFs to physical servers and for the logical links to physical links """
        # Numpy array of binary variables for the mapping of VNFs to physical servers (the only ones we need for now)
        self.phi_node = self.gpmodel.addMVar((len(self.sfc), len(self.vertices_P())), vtype=GRB.BINARY, name="phi_nodes")
        self.phi_link = self.gpmodel.addMVar((len(self.sfc)-1, len(self.edges_P())), vtype=GRB.BINARY, name="phi_link")

    def generate_mapping_constraints(self):
        # Each VNF must be mapped to exactly one physical server
        for v in range(len(self.sfc)):
            self.gpmodel.addConstr(
                gp.quicksum(self.phi_node[v, i] for i in range(len(self.vertices_P()))) == 1
            )
        # Flow conservation constraints for the logical links 
        for i_index, i in enumerate(self.vertices_P()):
            for vlink in range(len(self.logical_links)):
                self.gpmodel.addConstr(
                    gp.quicksum(
                        self.phi_link[vlink, self.physical_link_index[(i, j)]] 
                        - self.phi_link[vlink, self.physical_link_index[(j, i)]] 
                        for j_index, j in enumerate(self.physGraph[i])
                    )
                    == - self.phi_node[vlink, i_index] + self.phi_node[vlink+1, i_index] 
                    # last line to be modified when we'll have more complicated VNF graphs, for now it's ok since we have a linear SFC
                                                                                                        
                )

    def generate_availability_constraints(self):
        # Availability constraints for the physical servers only, access nodes excluded in the range
        for i_index in range(1, len(self.vertices_P())-1):
            # In terms of computing resource
            self.gpmodel.addConstr(
                gp.quicksum(
                    self.phi_node[v_index, i_index] * self.computing_requirements[v] 
                        for v_index, v in enumerate(self.sfc)
                ) <=  self.computing_availability[self.vertices_P()[i_index]]
            )
            # In terms of memory resource
            self.gpmodel.addConstr(
                gp.quicksum(
                    self.phi_node[v_index, i_index] * self.memory_requirements[v] 
                        for v_index, v in enumerate(self.sfc)
                ) <=  self.memory_availability[self.vertices_P()[i_index]]
            )

        # And in terms of bandwidth usage
        for i, j in self.physical_links:
            self.gpmodel.addConstr(
                gp.quicksum(
                    self.phi_link[v_link, self.physical_link_index[(i, j)]] * self.bandwidth_requirement[v_link] 
                        for v_link in range(len(self.logical_links))
                ) <= self.bandwidth_availability[self.physical_link_index[(i, j)]]
            )

    def generate_access_nodes_constraints(self):
        # First VNF must be mapped to the access node i1 and the last VNF must be mapped to the access node i4 (or whatever the last node is)
        self.gpmodel.addConstr(self.phi_node[0, 0] == 1)
        self.gpmodel.addConstr(self.phi_node[len(self.sfc)-1, len(self.vertices_P())-1] == 1)
        # and only those two VNFs can be mapped to the access nodes
        for v in range(1, len(self.sfc)-1):
            self.gpmodel.addConstr(self.phi_node[v, 0] == 0)
            self.gpmodel.addConstr(self.phi_node[v, len(self.vertices_P())-1] == 0)

    def energy_cost(self):
        self.Ce = gp.quicksum(
            self.energy_price[self.vertices_P()[i]] * gp.quicksum(
                self.phi_node[v, i] for v in range(len(self.sfc))
            ) for i in range(len(self.vertices_P()))
        )
        return self.Ce
    
    def objective_function(self):
        self.gpmodel.setObjective(self.energy_cost(), GRB.MINIMIZE)

    def compute_model(self):
        self.generate_mapping_variables()
        self.generate_mapping_constraints()
        self.generate_availability_constraints()
        self.generate_access_nodes_constraints()
        self.objective_function()

    def optimize(self):
            try:
                self.gpmodel.optimize()
                if self.gpmodel.Status == GRB.OPTIMAL:
                    for v in self.gpmodel.getVars():
                        print(f"{v.VarName} {v.X:g}")
                    print(f"Obj: {self.gpmodel.ObjVal:g}")
                elif self.gpmodel.Status == GRB.INFEASIBLE:
                    print("Model is infeasible")
                else:
                    print(f"Optimization finished with status {self.gpmodel.Status}")

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")
            except AttributeError:
                print("Encountered an attribute error")


if __name__ == "__main__":
    model = json_parser("model2")
    network_mapping = NetworkMapping(model)
    network_mapping.compute_model()
    network_mapping.optimize()
    print("Physical Links:", network_mapping.physical_links)
