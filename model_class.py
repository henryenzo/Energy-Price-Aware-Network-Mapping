""" PERSONAL NOTES AND COMMENTS
    This program is a simple implementation of the energy price aware network mapping problem for Infrastructure Providers (InP) to reduce their energy costs. The first implementation will take only into account a fraction of the constraints in my draft, and with a not-yet proper modeling of the physical and logical graphs, but it will be a good starting point to test the model and then we can start to add more constraints and a better modeling of the graphs.

    I recall having a program that generates random graphs, it will be useful to test the model with different topologies and different parameters but that's not the purpose of this first draft

    TODO : I have a question relative to the flow conservation constraint. Is it really a sum over each j like in Trung's paper "Accelerating Network Slice Embedding..." or only the neighbours ? I considered only the neighbors here because it doesn't make any sense to map logical links to non-existing physical links, but maybe the variable has a hidden role.

    Created on June 16th, 2026 by Enzo Henry
"""

""" ABOUT THIS BRANCH : myopic model 
    This is baseline 2 : we consider a time-variying energy price, but we don't consider the migration cost yet. We have W=S=1 so we don't anticipate, we just optimize the mapping at each time step. The model is still a MIP, but we have to solve it at each time step, which is not optimal but it's a first step. We will see later how to add the migration cost and the anticipation of the future energy price (that will be baseline 3).
    I may call this approach "greedy" as well in the comments but as I've seen in the literature, greedy is more about the algorithm than the model itself, and the correct term is "myopic" ^^
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import json
import graphviz
from pathlib import Path
from stochastic_engine import get_energy_prices_from_csv

from PIL import Image
import math

def json_parser(model_name: str, file_name = "test_models.json") -> dict:
    """
        Helper function to parse the json file containing the test models and return the parameters of the model as a dict.\n
        Arguments:  
        - model_name: str, the name of the model to parse \n
        - file_name:  str, the name of the json file containing the test models (default: "test_models.json" is the one I used for tests) \n

        Returns: dict containing every paramters for the model (physGraph, sfc, availability, requirements etc)
    """

    with open(file_name, 'r') as file:
        test_models = json.load(file)["test_models"]
    for model in test_models:
        if model["model_name"] == model_name:
            return model["model_parameters"]
    raise ValueError(f"Model '{model_name}' does not exist")


class NetworkMapping:
    def __init__(self, model: dict):
        """
            Constructor of the class, takes a dict as input containing the model parameters (physGraph, sfc, availability, requirements etc) and initializes the class attributes accordingly. \n
            The model dict is expected to be imported from the json file using the `json_parser` function. \n
            For example:
            ```
            model = json_parser("model1")
            network_mapping = NetworkMapping(model)
            ``` \n
        """
        self.N = 10
        self.gpmodel = gp.Model("mip1")
        self.optimized_flag = 0

        self.physGraph = model["physGraph"]
        self.physical_nodes = list(self.physGraph.keys())
        self.physical_nodes_index = {node: idx for idx, node in enumerate(self.physical_nodes)}
        self.access_nodes = model["access_nodes"]   # is a dict like {"v1": "i1", "v7": "i10"}
        self.physical_links = self.__generate_edges("physical")             # list of 2-list representing a physical link 
        self.physical_link_index = {tuple(edge): idx for idx, edge in enumerate(self.physical_links)}

        self.virtualGraph = model["virtualGraph"]
        self.virtual_nodes = list(self.virtualGraph.keys())
        self.virtual_nodes_index = {node: idx for idx, node in enumerate(self.virtual_nodes)}
        self.logical_links = self.__generate_edges("virtual")               # list of 2-list representing a logical link 
        self.logical_links_index = {tuple(edge): idx for idx, edge in enumerate(self.logical_links)}


        self.computing_availability = model["computing_availability"]
        self.memory_availability    = model["memory_availability"]
        self.bandwidth_availability = [model["bandwidth_availability_dict"][vertex][neighbor] for vertex, neighbor in self.physical_links]

        self.computing_requirements = model["computing_requirements"]       # dict of computing requirement for each VNF
        self.memory_requirements    = model["memory_requirements"]          # dict as well
        self.bandwidth_requirement  = [model["bandwidth_requirements_dict"][vertex][neighbor] for vertex, neighbor in self.logical_links]  

        # Energy price
        try:
            model_country_list = list(set(model["node_country"].values())) 
            self.price_per_country  = get_energy_prices_from_csv(csv_file_name="energy_prices.csv", time_slots=self.N,country_list=model_country_list, stride=4)
            
            self.energy_price = {node: list(np.round(np.array(self.price_per_country[country])/100, 2)) for node, country in model["node_country"].items()}
            # division by 100 because the energy price values are too high compared to usage/disposal cost of nodes
        except Exception as e:
            print(f"Could not fetch energy prices from CSV, using default values")
            self.energy_price       = model["energy_price"]                 # dict i1: [price_t1, price_t2, ..., price_tn] 

        self.CPU_usage_price        = model["CPU_usage_price"]
        self.memory_usage_price     = model["memory_usage_price"]
        self.bandwidth_usage_price  = model["bandwidth_usage_price"]
        self.node_disposal_price    = model["node_disposal_price"]

        # time aspects for the greedy model
        self.k = 0      # maximum is self.N - 1
        self.cost = []
        self.overall_cost = 0

        self.verbose = False
    
    def __generate_edges(self, graph="physical"):
        """ generates the edges of the graph obtained by BFS from i1 to last, as a list of 2-lists, each 2-list representing an edge """
        edges = []
        if graph == "physical":
            for vertex in self.physGraph:
                for neighbour in self.physGraph[vertex]:
                    if (neighbour, vertex) not in edges:
                        edges.append([vertex, neighbour])
        elif graph == "virtual":
            for vertex in self.virtualGraph:
                for neighbour in self.virtualGraph[vertex]:
                    if (neighbour, vertex) not in edges:
                        edges.append([vertex, neighbour])
        return edges

    def generate_mapping_variables(self):
        """ generates the mapping variables for the VNFs to physical servers and for the logical links to physical links """
        # Numpy array of binary variables for the mapping of VNFs to physical servers (the only ones we need for now)
        self.phi_node = self.gpmodel.addMVar((len(self.virtual_nodes), len(self.physical_nodes)), vtype=GRB.BINARY, name="phi_nodes")
        self.phi_link = self.gpmodel.addMVar((len(self.logical_links), len(self.edges_P())), vtype=GRB.BINARY, name="phi_link")
        # node activation variables, sigma_i = 1 if at least one VNF is mapped to node i, 0 otherwise
        self.sigma = self.gpmodel.addMVar((len(self.physical_nodes),), vtype=GRB.BINARY, name="sigma")
        # migration variables, xi_v,i = 1 if VNF v is migrated to node i, 0 otherwise
        #self.xi = self.gpmodel.addMVar((len(self.virtual_nodes), len(self.physical_nodes)), vtype=GRB.BINARY, name="xi")

    def generate_mapping_constraints(self):
        """ generates the mapping constraints on phi_node and phi_link """
        # Each VNF must be mapped to exactly one physical server
        for v in range(len(self.virtual_nodes)):
            self.gpmodel.addConstr(
                gp.quicksum(self.phi_node[v, i] for i in range(len(self.physical_nodes))) == 1
            )
        # Flow conservation constraints for the logical links 
        for i_index, i in enumerate(self.physical_nodes):
            for vlink_index, (v, w) in enumerate(self.logical_links):
                self.gpmodel.addConstr(
                    gp.quicksum(
                        self.phi_link[vlink_index, self.physical_link_index[(i, j)]] 
                        - self.phi_link[vlink_index, self.physical_link_index[(j, i)]] 
                        for j_index, j in enumerate(self.physGraph[i])
                    )
                    == self.phi_node[self.virtual_nodes_index[v], i_index] - self.phi_node[self.virtual_nodes_index[w], i_index]                
                )
    
    def generate_node_activation_constraints(self):
        # sigma_i = 1 if at least one VNF is mapped to node i, 0 otherwise
        # addGenConstrOr is a Gurobi function that takes the logical OR of a list of binary variables, here all the mapped VNFs to node i
        for i_index in range(len(self.physical_nodes)):
            self.gpmodel.addGenConstrOr(        
                self.sigma[i_index],
                [self.phi_node[v, i_index] for v in range(len(self.virtual_nodes))]
            )

    def generate_availability_constraints(self):
        # Availability constraints for the physical servers only, access nodes excluded in the range
        for i_index, i in enumerate(self.physical_nodes):
            if i not in self.access_nodes: # just erase this line to apply the constraints to access nodes as well
                # In terms of computing resource
                self.gpmodel.addConstr(
                    gp.quicksum(
                        self.phi_node[v_index, i_index] * self.computing_requirements[v] 
                            for v_index, v in enumerate(self.virtual_nodes)
                    ) <=  self.computing_availability[self.physical_nodes[i_index]]
                )
                # In terms of memory resource
                self.gpmodel.addConstr(
                    gp.quicksum(
                        self.phi_node[v_index, i_index] * self.memory_requirements[v] 
                            for v_index, v in enumerate(self.virtual_nodes)
                    ) <=  self.memory_availability[self.physical_nodes[i_index]]
                )

        # And in terms of bandwidth usage
        for i, j in self.physical_links:
            self.gpmodel.addConstr(
                gp.quicksum(
                    self.phi_link[vlink_index, self.physical_link_index[(i, j)]] * self.bandwidth_requirement[vlink_index] 
                        for vlink_index, (v,w) in enumerate(self.logical_links)
                ) <= self.bandwidth_availability[self.physical_link_index[(i, j)]]
            )

    def generate_access_nodes_constraints(self):
        # First VNF must be mapped to the first access node and the last VNF must be mapped to the last access node 
        for access_node in self.access_nodes.items():
             # First VNF must be mapped to the first access node and the last VNF must be mapped to the last access node 
            self.gpmodel.addConstr( 
                self.phi_node[
                    self.virtual_nodes_index[access_node[0]], 
                    self.physical_nodes_index[access_node[1]]
                ] == 1
            )
            # and only those two VNFs can be mapped to the access nodes
            for v_index, v in enumerate(self.virtual_nodes):
                if v not in self.access_nodes.keys():
                    self.gpmodel.addConstr(self.phi_node[v_index, self.physical_nodes_index[access_node[1]]] == 0)

    def energy_cost(self): 
        # depends on k 
        self.Ce = gp.quicksum(
            self.energy_price[self.physical_nodes[i]][self.k] * gp.quicksum(
                self.phi_node[v, i] for v in range(len(self.virtual_nodes))
            ) for i in range(len(self.physical_nodes))
        )
        return self.Ce
    
    def disposal_cost(self):
        self.Cf = gp.quicksum(
            self.node_disposal_price[self.physical_nodes[i]] * self.sigma[i] for i in range(len(self.physical_nodes))
        )
        return self.Cf
    
    def usage_cost(self):
        self.Cr = gp.quicksum(
            gp.quicksum(
                self.CPU_usage_price[self.physical_nodes[i]] * self.phi_node[v, i] * self.computing_requirements[self.virtual_nodes[v]] 
                for v in range(len(self.virtual_nodes))
            ) 
            + gp.quicksum(
                self.memory_usage_price[self.physical_nodes[i]] * self.phi_node[v, i] * self.memory_requirements[self.virtual_nodes[v]] 
                for v in range(len(self.virtual_nodes))
            ) 
            + gp.quicksum(
                self.bandwidth_usage_price[self.physical_nodes[i]][self.physical_nodes[j]] 
                * self.phi_link[v_link, self.physical_link_index[(self.physical_nodes[i], self.physical_nodes[j])]] 
                * self.bandwidth_requirement[v_link] 
                for v_link in range(len(self.logical_links)) 
                for j in range(len(self.physical_nodes)) if (self.physical_nodes[i], self.physical_nodes[j]) in self.physical_links
            )
            for i in range(len(self.physical_nodes))
        )
        return self.Cr
    
    def link_usage_cost(self):
        """
            I'm not sure about this one yet, in my draft I don't take it into account but it will force the model to take the shortest path for the logical links, which is a good thing I guess.\n
            We take a fix cost of 0.1 for each link used
        """
        self.Cl = gp.quicksum(
            0.1 * self.phi_link[v_link, self.physical_link_index[(ij[0], ij[1])]]
            for v_link in range(len(self.logical_links))
            for ij in self.physical_links
        )
        return self.Cl
            

    def objective_function(self):
        """
            Objective function of the problem : $E_{InP} = C_e + C_r + C_f + C_l$

            where:
            - C_e is the energy cost (variable cost), \n
            - C_r is the resource usage cost (variable cost), \n
            - C_f is the disposal cost (fix cost), \n
            - C_l is the link usage cost (this one doesn't appear in my paper, it is to make the model choose the shortest paths). \n
            The objective function is to **minimize** the total cost $E_{InP}$.
        """
        self.gpmodel.setObjective(self.energy_cost() + self.usage_cost() + self.disposal_cost() + self.link_usage_cost(), GRB.MINIMIZE)

    def compute_model(self):
        """
            Computes the model by generating the mapping variables, the mapping constraints, the availability constraints, the access nodes constraints and the objective function. \n
            This method must be called before `self.optimize()` in order to compute the model and optimize it. 
        """
        self.generate_mapping_variables()
        self.generate_mapping_constraints()
        self.generate_node_activation_constraints()
        self.generate_availability_constraints()
        self.generate_access_nodes_constraints()
        self.objective_function()

    def update_model(self):
        """
            Updates the model for the next time slot k+1 by updating the energy price of the physical nodes. \n
            This method allows us to only update what changes temporally, without having to recompute the whole model from scratch. \n
            Basically only the energy price changes in this myopic model
        """
        self.k += 1
        self.objective_function()


    def optimize(self):
        """
            Optimizes the model and prints the results. \n
            Requires the model to have been computed with `self.compute_model()` first.\n
            If the model is optimal, it will also plot the physical graph (see `self.plot_graph()`),\n
            if it's not, will print the status of the optimization and the reason why it failed (infeasible, unbounded, etc.)\n
            Exceptions are handled but not extensively
        """
        try:
            self.gpmodel.optimize()
            if self.gpmodel.Status == GRB.OPTIMAL:
                if self.verbose:
                    for v in self.gpmodel.getVars():
                        print(f"{v.VarName} {v.X:g}")
                print(f"Obj: {self.gpmodel.ObjVal:g}")
                self.optimized_flag = 1
                # self.plot_graph()
            elif self.gpmodel.Status == GRB.INFEASIBLE:
                print("Model is infeasible")
            else:
                print(f"Optimization finished with status {self.gpmodel.Status}")

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")
        except AttributeError:
            print("Encountered an attribute error")

    def run(self):
        """
            Runs the model by computing it and optimizing it for each time slot k.
        """
        # first iteration (k=0)
        self.compute_model()
        self.optimize()
        self.plot_graph(graph_name=f"physical_graph_k{self.k}")
        self.cost.append(self.gpmodel.ObjVal)
        self.overall_cost += self.gpmodel.ObjVal
        for k in range(1, min(len(self.energy_price[self.physical_nodes[0]])), self.N): # fail-safe to avoid going out of bounds if the energy price list is shorter than N, maybe will it be better to integrate this directly into the constructor ?
            self.update_model()
            self.optimize()
            self.plot_graph(graph_name=f"physical_graph_k{self.k}")
            self.cost.append(self.gpmodel.ObjVal)
            self.overall_cost += self.gpmodel.ObjVal
        print(f"Cost for each time slot k: {self.cost}")
        print(f"Overall cost for the whole time horizon: {self.overall_cost}")

    
    def plot_graph(self, graph_name="physical_graph"):
        """ 
            Plots the physical graph highlighting the mapping, energy price and resource utilization.\n
            This method uses the *neato* layout engine of `graphviz` python library, for easier visualization of the graph. \n
            The nodes are colored in red if they are servers hosting at least one VNF, and in blue if they are access nodes. Label: 
            - c : computing resource usage / availability (on red and blue nodes only)
            - m : memory resource usage / availability (on red and blue nodes only)
            - e : energy price (on all nodes) \n
            The edges are colored in red if they are used to map at least one logical link, and indicate their bandwidth usage / availability. \n
            Saves the plot as  `./plots/physical_graph.svg`\n

            Argument : graph_name (str) : name of the graph and the file to save, default is "physical_graph"
        """
        assert self.optimized_flag, "The model must have been optimized in order to generate the graph"
        plots_dir = Path(__file__).resolve().parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        g = graphviz.Digraph(
            graph_name,                     # name of the graph
            filename=graph_name,            # name of the file
            engine='neato',                 # layout engine (neato produces )
            format='svg',                   # output format
            #rankdir='LR',                  # direction of the graph (LR = left to right), but this parameternot supported by neato
        )
        
        g.attr(overlap='false')
        g.attr(sep='+0')
        for i, j in self.physical_links:
            i_used  =   [self.virtual_nodes[v] for v in range(len(self.virtual_nodes)) if self.phi_node.X[v, self.physical_nodes_index[i]]]
            j_used  =   [self.virtual_nodes[v] for v in range(len(self.virtual_nodes)) if self.phi_node.X[v, self.physical_nodes_index[j]]]
            ij_used =   [self.logical_links[v_link] for v_link in range(len(self.logical_links)) if self.phi_link.X[v_link, self.physical_link_index[(i, j)]]]

            link_label = f"{self.phi_link.X[:, self.physical_link_index[(i, j)]] @ self.bandwidth_requirement[:]}/{self.bandwidth_availability[self.physical_link_index[(i, j)]]}"

            node_i_compute = f" \n c: {sum(self.phi_node.X[index_v, self.physical_nodes_index[i]] * self.computing_requirements[v] for index_v, v in enumerate(self.virtual_nodes))}/{self.computing_availability[i]}"
            node_i_memory = f" \n m: {sum(self.phi_node.X[index_v, self.physical_nodes_index[i]] * self.memory_requirements[v] for index_v, v in enumerate(self.virtual_nodes))}/{self.memory_availability[i]}"
            node_i_energy_price = f" \n e: {self.energy_price[i][self.k]}"
            node_i_label = f"{i} ({i_used})" + node_i_compute + node_i_memory + node_i_energy_price
            
            node_j_compute = f" \n c: {sum(self.phi_node.X[index_v, self.physical_nodes_index[j]] * self.computing_requirements[v] for index_v, v in enumerate(self.virtual_nodes))}/{self.computing_availability[j]}"
            node_j_memory = f" \n m: {sum(self.phi_node.X[index_v, self.physical_nodes_index[j]] * self.memory_requirements[v] for index_v, v in enumerate(self.virtual_nodes))}/{self.memory_availability[j]}"
            node_j_energy_price = f" \n e: {self.energy_price[j][self.k]}"
            node_j_label = f"{j} ({j_used})" + node_j_compute + node_j_memory + node_j_energy_price

            if len(ij_used) > 0:
                g.edge(i, j, label=link_label, fontsize='10', color='red', fontcolor='red')
            else:
                g.edge(i, j, label=str(self.bandwidth_availability[self.physical_link_index[(i, j)]]), fontsize='10')
                g.node(i, label=f"{i} \n e: {self.energy_price[i][self.k]}", fontsize='10')
                g.node(j, label=f"{j} \n e: {self.energy_price[j][self.k]}", fontsize='10')
            if len(i_used) > 0:
                g.node(i, label=node_i_label, fontsize='10', color='red', fontcolor='red')
            if len(j_used) > 0:
                g.node(j, label=node_j_label, fontsize='10', color='red', fontcolor='red')
                
        for access_node in self.access_nodes.values():
            g.node(access_node, color='blue', fontcolor='blue')

        g.attr(label=f"k={self.k}", labelloc="t", labeljust="l", fontsize="14", fontcolor="black")
        g.render(directory=str(plots_dir), cleanup=True)

    def plot_all_graphs(self, k_values, n_cols=5):
        # this function was generated by Claude AI to help me plot all the graphs for each time slot k in a grid, and save it as a single image
        # but I am not satisfied with the result (all_timeslots), as it is barely reeadable and rasterized. Will I need to display them all on the same figure for my paper ?
        paths = [f"plots/physical_graph_k{k}.png" for k in k_values]
        images = [Image.open(p) for p in paths]
        w, h = max(im.width for im in images), max(im.height for im in images)
        n_rows = math.ceil(len(images) / n_cols)

        grid = Image.new("RGB", (w * n_cols, h * n_rows), "white")
        for idx, im in enumerate(images):
            row, col = divmod(idx, n_cols)
            grid.paste(im, (col * w, row * h))

        grid_path = Path(__file__).resolve().parent / "plots" / "all_timeslots.png"
        grid.save(grid_path)
        return grid_path


if __name__ == "__main__":
    #model = json_parser("model3.2")
    model = json_parser("nobel-eu")
    network_mapping = NetworkMapping(model)
    network_mapping.run()
    print("Physical Links:", network_mapping.physical_links)
    print("Logical links: ", network_mapping.logical_links)
    # network_mapping.plot_all_graphs(k_values=range(len(network_mapping.energy_price[network_mapping.physical_nodes[0]])), n_cols=5)
