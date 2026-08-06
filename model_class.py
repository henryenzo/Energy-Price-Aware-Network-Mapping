""" PERSONAL NOTES AND COMMENTS
    This program is a simple implementation of the energy price aware network mapping problem for Infrastructure Providers (InP) to reduce their energy costs. The first implementation will take only into account a fraction of the constraints in my draft, and with a not-yet proper modeling of the physical and logical graphs, but it will be a good starting point to test the model and then we can start to add more constraints and a better modeling of the graphs.

    I recall having a program that generates random graphs, it will be useful to test the model with different topologies and different parameters but that's not the purpose of this first draft

    TODO : I have a question relative to the flow conservation constraint. Is it really a sum over each j like in Trung's paper "Accelerating Network Slice Embedding..." or only the neighbours ? I considered only the neighbors here because it doesn't make any sense to map logical links to non-existing physical links, but maybe the variable has a hidden role.

    TODO : multiple SFCs to implement everywhere maybe, for now it's only on the delay constraint. 

    TODO : maybe adapt the prediction window thig to give more importance to early gains, like a discount factor. Because one of the main problems we observe is the fact that W=5 is not always the most efficient because what is optimized is a mean value for the whole window. If another migration is to be done, basically the gain that we were supposed to have is lost because we just changed the placement of the VNFs. So maybe we can consider a discount factor for the future costs, like a geometric series with a discount factor of 0.9 or 0.8, so that the optimizer will prefer to have a lower cost at the beginning of the window rather than at the end.

    TODO : find a better day to conduct the tests, and a larger spectrum too have sometimes where migration is worth and some where it's absolutely not. 

    Created on June 16th, 2026 by Enzo Henry
"""

""" ABOUT THIS BRANCH : foresighted model 
    This is baseline 3 : we consider time-varying parameters, and a migration cost (for now fixed). We will consider W the time window of estimated (deterministic here) parameters, and S the number of time steps on which we will optimize. 
    Multiple approaches are possible, considering a space-time variable extension could be too complex and cause scalability issues, so maybe we will consider a pool of "migration-candidate" VNFs and only keep the mapping variables for those VNFs, and then we will have to consider a migration cost for each VNF that is migrated from one node to another. 

    other ways to do it : 
    - relax the integer constraint on the interval [k+S, k+W] so that we can have a continuous variable for the mapping of the VNFs, so the complexity goes from linear to the time window W to logarithmic (I think it was in the chapter 7 of Wolsey's "Integer Programming" book)
    - warm start the model with the previous mapping allows the optimizer to converge faster and avoid instability (migration cost function will penalize the model for migrating VNFs unnecessarily anyways)
    - warm start using Machine Learning. Supervised learning to predict the mapping of the VNFs for the next time step, and then use this prediction as a warm start for the optimization model (I need to finish reading Nair et al. 2021 (arXiv:2012.13349v3)). Basically a binary classifier on each VNF. For this I will need a pretty good graph dataset or a good graph generator (in that case I will most certainly use NetworkX as Trung and Michel told me). I'm working on the stochastic engine module to generate all the parameters for the model, this won't take too long imo.

    Actually we will do as such : we consider the parameters on a window W of observation and we'll decide on only S=1 time step, so with only one set of variables for the mapping of the VNFs.
    For that we'll be needing a few additional functions total_window_cost, migration_cost, total_window_constraints, and total_window_objective_function.
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
    def __init__(self, model: dict, W: int = 1, S: int = 1, N: int = 10, time_stride: int =1, offset: int = 0, prices_csv: str = "energy_prices.csv", graphviz_output_dir: str = "plots/graphs"):
        """
            Constructor of the class, takes a dict as input containing the model parameters (physGraph, sfc, availability, requirements etc) and initializes the class attributes accordingly. \n
            The model dict is expected to be imported from the json file using the `json_parser` function. \n
            For example:
            ```
            model = json_parser("model1")
            network_mapping = NetworkMapping(model)
            ``` \n
        """
        self.N = N
        self.W = W # time window of observation for the foresighted model
        self.S = S # time steps to optimize for the foresighted model, we can only set it to one for now, I don't even think a higher value would be useful.

        # Gurobi model hyperparameters
        self.gpmodel = gp.Model("mip1")
        #self.gpmodel.Params.MIPGap = 1e-9          # too restrictive
        #self.gpmodel.Params.MIPGapAbs = 1e-12      # too restrictive
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
            self.price_per_country  = get_energy_prices_from_csv(csv_file_name=prices_csv, time_slots=self.N+self.W,country_list=model_country_list, stride=time_stride, starting_index=offset)
            
            self.energy_price = {node: list(np.array(self.price_per_country[country])) for node, country in model["node_country"].items()}
            # division by 100 --> deleted now, shouldn't have lasted that long
            print(f"Energy prices fetched from ENTSO-E CSV for the countries : {model_country_list}")
            print(len(self.energy_price[self.physical_nodes[0]]), "time slots fetched for each node") 
        except Exception as e:
            print(f"Could not fetch energy prices from CSV file '{prices_csv}'")
            raise e
            
        self.CPU_usage_price        = model["CPU_usage_price"]
        self.memory_usage_price     = model["memory_usage_price"]
        self.bandwidth_usage_price  = model["bandwidth_usage_price"]
        self.node_disposal_price    = model["node_disposal_price"]

        self.links_distance_dict    = model["links_distance_dict"]   # in hundreds of km

        # time aspects for the greedy model
        self.k = 0      # maximum is self.N - 1
        self.cost = []
        self.overall_cost = 0
        self.placement = []     # list of dicts, each dict for 1 time slot k, with the mapping of VNFs to physical servers
        self.migrations = []    # list of number of migrations for each time slot k

        self.verbose = False

        self.max_delay = 0.4 # s (400ms SLA per SFC, so might need to adjust when I put more SFCs) TODO: make it scale with the number of SFCs
        #self.max_delay = model["max_delay"]
        self.migration_downtime = 0.124 # in seconds. Comes from Liu2011 (Dbench benchmark) : closest workload to a NAT/FW/TM VNF apparently (otherwise it's incomparable)

        # predictive model
        self.prev_phi_node = np.array([])   # this init serves no purpose, it will be updated after the first optimization but I need to have an overview
        self.prev_phi_link = np.array([])   # same for the links
        self.prev_sigma = np.array([])      # same for the node activation variables
        self.prev_xi = np.array([])         # same for the migration variables
        try:
            self.migration_energy_cost = model["migration_energy_cost"] # clearly this will be a dict v1: cost, v2: cost, ... 
        except:
            self.migration_energy_cost = {v: 1 for v in self.virtual_nodes} # default value of 1 for each VNF, 

        # if we have multiple SFCs
        # (still in development)
        if isinstance(self.virtualGraph, list):
            self.virtual_nodes = []
            self.logical_links = []
            for sfc in self.virtualGraph:
                self.virtual_nodes += list(sfc.keys())
                self.logical_links += self.__generate_edges("virtual")
            self.virtual_nodes_index = {node: idx for idx, node in enumerate(self.virtual_nodes)}
            self.logical_links_index = {tuple(edge): idx for idx, edge in enumerate(self.logical_links)}

        self.graphviz_output_dir = graphviz_output_dir
    
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

    def generate_variables(self):
        """ generates the mapping variables for the VNFs to physical servers and for the logical links to physical links """
        # Numpy array of binary variables for the mapping of VNFs to physical servers (the only ones we need for now)
        self.phi_node = self.gpmodel.addMVar((len(self.virtual_nodes), len(self.physical_nodes)), vtype=GRB.BINARY, name="phi_nodes")
        self.phi_link = self.gpmodel.addMVar((len(self.logical_links), len(self.physical_links)), vtype=GRB.BINARY, name="phi_link")
        # node activation variables, sigma_i = 1 if at least one VNF is mapped to node i, 0 otherwise
        self.sigma = self.gpmodel.addMVar((len(self.physical_nodes),), vtype=GRB.BINARY, name="sigma")
        # migration variables, xi_v,i = 1 if VNF v is migrated to node i, 0 otherwise
        self.xi = self.gpmodel.addMVar((len(self.virtual_nodes), len(self.physical_nodes)), vtype=GRB.BINARY, name="xi")

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

    def generate_migration_constraints(self):
        """
            Subsection 3.3 of my paper draft, the migration constraints are as follows : \n
        """
        self.migration_constrs = []
        for v in range(len(self.virtual_nodes)):
            for i in range(len(self.physical_nodes)):
                c1 = self.gpmodel.addConstr(
                    self.xi[v, i] >= self.phi_node[v, i] - self.prev_phi_node[v, i]
                )
                c0 = self.gpmodel.addConstr(self.xi[v, i] <= self.phi_node[v, i])

                c2 = self.gpmodel.addConstr(
                    self.xi[v, i] <= 1 - self.prev_phi_node[v, i]
                )
                self.migration_constrs += [c1, c0, c2]
            c3 = self.gpmodel.addConstr(
                gp.quicksum(self.xi[v, i] for i in range(len(self.physical_nodes))) <= 1
            )
            self.migration_constrs.append(c3)

    def generate_delay_constraints(self):
        """
            Generates the constraints on the total delay introduced by the routing of the SFC onto the physical network.\n
            The formula that I'll use is : $\sum_{vw\in logical_links of s} \sum_{ij\in physical_links} phi_link_{vw:ij} * d_{ij} <= max_delay(s)  forall SFC s$ \n
            for now, fix delay but then I'll enlarge it to be an SFC parameter. TODO !!!
            If this constraint isn't verified, the optimizer will cancel the mapping, so TODO make it non destructive

            Any estimation of the delay required for the transmission of a packet based on the distance between nodes will be rough, because it depends on the actual network topology and traffic conditions. We may consider a reasonable and usual value of a 0.5ms delay per 100km of otpical fiber + 2ms of switch per hop + more if hosting a VNF
        """
        delay_per_100km = 0.0005 # 0.5ms per 100km of optical fiber
        delay_per_hop = 0.001 # 1ms per hop (switching delay)
        delay_per_VNF = 0.002 # 2ms per VNF hosting (processing delay, idk if it's realistic)

        if isinstance(self.virtualGraph, list): #if we differenciate SFCs
            for sfc in self.virtualGraph:
                self.gpmodel.addConstr(
                    gp.quicksum(
                        self.phi_link[self.logical_links_index[(v, w)], ij_index] *  self.links_distance_dict[ij[0]][ij[1]] * delay_per_100km
                        for ij_index, ij in enumerate(self.physical_links)
                        for v, w in sfc.items()
                    )
                    + gp.quicksum( # migration downtime
                        self.migration_downtime * self.xi[self.virtual_nodes_index[v], i]
                        for v in sfc.keys()
                        for i in range(len(self.physical_nodes))
                    ) <= self.max_delay
                )
        else:
            self.gpmodel.addConstr(
                gp.quicksum(
                    self.phi_link[vw_index, ij_index] * (
                        self.links_distance_dict[ij[0]][ij[1]] * delay_per_100km
                        + delay_per_hop # corresponds to the switching delay at the physical nodes
                        + self.phi_node[self.virtual_nodes_index[vw[1]], self.physical_nodes_index[ij[1]]] * delay_per_VNF
                    )
                    for ij_index, ij in enumerate(self.physical_links)
                    for vw_index, vw in enumerate(self.logical_links)
                )
                + gp.quicksum( 
                    self.migration_downtime * self.xi[v_index, i]
                    for v_index in range(len(self.virtual_nodes))
                    for i in range(len(self.physical_nodes))
                ) <= self.max_delay
            )

    def energy_cost(self, k=None): 
        """
            REALISTIC ENERGY COST FUNCTION :\n
            From the real energy price in Europe, the current CPU load, and considering the servers to use Intel i7 14th gen CPUs, we can estimate the energy cost for each physical server at each time slot k: \n
            $C_e(k) = \sum_{i\in physical_nodes} energy_price_i(k) * P_i(k)$ \n
            with:\n
            $P_i(k) = P_idle + (P_max - P_idle) * CPU_usage_i(k)$ \n
            $CPU_usage_i(k) = \sum_{v\in VNFs} phi_node_{v,i}(k) * computing_requirement_v / computing_availability_i$ \n
        """
        # we used to depend on a parameter from the json but now I use Intel i7 14th gen specifications and CPU usage
        # depends on k 
        if k is None:
            k = self.k
        # self.Ce = gp.quicksum(
        #     self.energy_price[self.physical_nodes[i]][k] * gp.quicksum(
        #         self.phi_node[v, i] for v in range(len(self.virtual_nodes))
        #     ) for i in range(len(self.physical_nodes))
        # )
        Watts_over_15min_to_MWh = 1/1000000 * 900/3600 
        P_idle = 65 # Watts
        P_max = 219 # Watts
        CPU_usage = lambda i: gp.quicksum(
            self.phi_node[v, i] * self.computing_requirements[self.virtual_nodes[v]] 
            for v in range(len(self.virtual_nodes))
        ) / self.computing_availability[self.physical_nodes[i]]
        Power = lambda i: P_idle + (P_max - P_idle) * CPU_usage(i)
        self.Ce = gp.quicksum(
            self.sigma[i] * self.energy_price[self.physical_nodes[i]][k] * Watts_over_15min_to_MWh * Power(i)
            for i in range(len(self.physical_nodes))
        ) 
        return self.Ce
    
    def disposal_cost(self, k=None):
        if k is None:
            k = self.k # but unused fot the moment
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
    
    def link_delay(self):
        delay_per_100km = 0.0005 # 0.5ms per 100km of optical fiber
        delay_per_hop = 0.001 # 1ms per hop (switching delay)
        delay_per_VNF = 0.002 # 2ms per VNF hosting (processing delay, idk if it's realistic)
        self.total_link_delay = gp.quicksum(
                self.phi_link[vw_index, ij_index] * (
                    self.links_distance_dict[ij[0]][ij[1]] * delay_per_100km
                    + delay_per_hop # corresponds to the switching delay at the physical nodes
                    + self.phi_node[self.virtual_nodes_index[vw[0]], self.physical_nodes_index[ij[0]]] * delay_per_VNF
                )
                for ij_index, ij in enumerate(self.physical_links)
                for vw_index, vw in enumerate(self.logical_links)
            )
        return self.total_link_delay

    def link_delay_cost(self, coefficient=0.0000001):
        """
           This will be our fictious cost to reduce delay. This should not appear in the effective cost (effective_cost_at_k) but will be in the objective function to reduce the delay
           It is supposed to be negligible compared to the other costs, but it will be used to make the optimizer choose the shortest paths for the logical links, and thus simplify the final graph (many path are available). \n
        """
        self.Cl = coefficient * self.link_delay()
        # return gp.LinExpr(0)
        return self.Cl
    
    def migration_cost(self, coefficient=1):
        """
            REALISTIC MIGRATION COST FUNCTION :\n
            Migration cost for the foresighted model, taking into account the migration of VNFs from one physical server to another. \n
            Here, I consider the cost to be the sum of the energy on each of the origin AND destination servers, multiplied by the energy price of course

            For now, I consider a fix value from Liu2011 which hopefully is still relevant. I use 375J of energy per server to migrate a VNF (750J total for 600MBps throughput, split evenly between origin and destination), and I multiply it by the energy price of the origin and destination servers. \n
        """
        Joules_to_MWh = 1/1000000 * 900/3600 # why 900 again ? it's just Joules so I'm guessing 900 should just disappear
        Watts_over_15min_to_MWh = 1/1000000 * 900/3600
        P_idle = 65 # Watts
        fix_migration_energy = 375 # Joules, Liu2011 (750J total split per server)
        if self.k == 0:
            self.Cm = gp.LinExpr(0) # no migration cost for the first time slot
            return self.Cm

        self.Cm = gp.quicksum(
            coefficient * Joules_to_MWh * fix_migration_energy 
            * (self.energy_price[i][self.k] + self.energy_price[j][self.k])
            * self.xi[self.virtual_nodes_index[v], self.physical_nodes_index[j]]
            * self.prev_phi_node[self.virtual_nodes_index[v], self.physical_nodes_index[i]]
            for v in self.virtual_nodes # for each VNF
            for i in self.physical_nodes # for the origin node
            for j in self.physical_nodes # for the destination node
        )+ gp.quicksum(
            self.energy_price[self.physical_nodes[i]][self.k] * Watts_over_15min_to_MWh * P_idle * self.xi[v, i]
            for v in range(len(self.virtual_nodes))
            for i in range(len(self.physical_nodes))
        )
        return self.Cm

    def total_window_cost(self):
        """
            Total cost over the time window W, for the foresighted model. \n
            This function will be used to compute the total cost over the time window W, and will be used in the objective function\n
        """
        if self.W == 0: # just one time slot, no migration cost
            self.total_cost = self.energy_cost(self.k) + self.link_delay_cost()
            return self.total_cost
        
        self.total_cost = gp.quicksum(
            self.energy_cost(k)  + self.link_delay_cost()
            # + self.usage_cost() + self.disposal_cost(k)
            for k in range(self.k, self.k + self.W)
        ) + self.migration_cost()
        return self.total_cost
    
    def total_window_objective_function(self):
        self.gpmodel.setObjective(self.total_window_cost(), GRB.MINIMIZE)
    
    def effective_cost_at_k(self):
        """
            Cost at time slot k, used to save the cost at each time slot and display the cumulative cost at the end of the optimization process. \n
        """
        if self.W == 0: # no migration cost
            return self.energy_cost().getValue() 
        
        # cost_k = self.energy_cost() + self.usage_cost() + self.disposal_cost() + self.migration_cost()
        cost_k = self.energy_cost() + self.migration_cost()  
        return cost_k.getValue()
    
    def individual_costs_at_k(self):
        """
            Individual effective costs at time slot k, for energy, migration, usage, and disposal, + the delay
        """
        if self.W == 0: # no migration cost 
            return {
                "energy_cost": self.energy_cost().getValue(),
                "link_delay": self.link_delay().getValue(),
                "link_delay_cost": self.link_delay_cost().getValue()
            }

        cost_k = {
            "energy_cost": self.energy_cost().getValue(),
            # "usage_cost": self.usage_cost().getValue(),
            # "disposal_cost": self.disposal_cost().getValue(),
            "link_delay": self.link_delay().getValue(),   # coeff =1 so we have the real delay in ms
            "link_delay_cost": self.link_delay_cost().getValue(),
            "migration_cost": self.migration_cost().getValue()
        }
        return cost_k

    def objective_function(self):
        """
            DEPRECATED : use total_window_objective_function instead, this one is for the myopic model only. \n
            Objective function of the problem : $E_{InP} = C_e + C_r + C_f + C_l$

            where:
            - C_e is the energy cost (variable cost), \n
            - C_r is the resource usage cost (variable cost), \n
            - C_f is the disposal cost (fix cost), \n
            - C_l is the link usage cost (this one doesn't appear in my paper, it is to make the model choose the shortest paths). \n
            The objective function is to **minimize** the total cost $E_{InP}$.
        """
        self.gpmodel.setObjective(self.energy_cost() + self.usage_cost() + self.disposal_cost() + self.link_delay_cost(), GRB.MINIMIZE)

    def warm_start(self):
        """
            Warm start the model with the previous mapping of the VNFs to physical servers and logical links to physical links. \n
            This method is useful to speed up the optimization process, especially for the foresighted model where we have a time window W of observation and we want to optimize for S time steps. \n
            The warm start is done by setting the initial values of the mapping variables to the previous values, and setting the initial values of the migration variables to 0 (no migration at the beginning).
        """
        if self.prev_phi_node.size > 0:
            self.phi_node.Start = self.prev_phi_node
        if self.prev_phi_link.size > 0:
            self.phi_link.Start = self.prev_phi_link
        if self.prev_sigma.size > 0:
            self.sigma.Start = self.prev_sigma
        # deleted xi init because it doesn't make any sense to warm start the migration variables

    
    def compute_model(self):
        """
            Computes the model by generating the mapping variables, the mapping constraints, the availability constraints, the access nodes constraints and the objective function. \n
            This method must be called before `self.optimize()` in order to compute the model and optimize it. 
        """
        self.generate_variables()
        self.generate_mapping_constraints()
        self.generate_node_activation_constraints()
        self.generate_availability_constraints()
        self.generate_access_nodes_constraints()
        for v_index in range(len(self.virtual_nodes)):
            for i in range(len(self.physical_nodes)):
                self.gpmodel.addConstr(self.xi[v_index, i] == 0)
        self.generate_delay_constraints()
        self.total_window_objective_function()  # modified

    def update_model(self):
        """
            Updates the model for the next time slot k+1 by updating the energy price of the physical nodes. \n
            This method allows us to only update what changes temporally, without having to recompute the whole model from scratch. \n
            Basically only the energy price changes in this myopic model
        """
        self.prev_phi_link = self.phi_link.X.copy()
        self.prev_phi_node = self.phi_node.X.copy()
        self.prev_sigma = self.sigma.X.copy()
        self.prev_xi = self.xi.X.copy()
        self.k += 1
        if hasattr(self, "migration_constrs"): # constraints cleanup
            for c in self.migration_constrs:
                self.gpmodel.remove(c)
        self.generate_migration_constraints()
        self.total_window_objective_function()
        self.warm_start()


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
        if self.W == 0:
            print("W=0 => running without migration, only optimizing for the current time slot k=0")
            self.run_nomig()
            return
        
        # first iteration (k=0)
        self.gpmodel.Params.OutputFlag = int(self.verbose)
        self.compute_model()
        self.optimize()
        self.plot_graph(graph_name=f"physical_graph_k{self.k:02d}")
        cost_k = self.effective_cost_at_k()
        individual_costs_k = self.individual_costs_at_k()
        self.cost.append({"effective_cost": cost_k, **individual_costs_k})
        self.placement.append({v: i for v in self.virtual_nodes for i in self.physical_nodes if self.phi_node.X[self.virtual_nodes_index[v], self.physical_nodes_index[i]] == 1})
        self.migrations.append(0) # no migration for the first time slot
        self.overall_cost += cost_k
        for k in range(1, min(len(self.energy_price[self.physical_nodes[0]]), self.N)): # fail-safe to avoid going out of bounds if the energy price list is shorter than N, maybe will it be better to integrate this directly into the constructor ?
            self.gpmodel.Params.OutputFlag = int(self.verbose)
            print(f"\n\nTime slot k={k} : ")
            print(f"Hyperparameters : W={self.W}, S={self.S}, N={self.N}")
            self.update_model()
            self.optimize()
            self.plot_graph(graph_name=f"physical_graph_k{self.k:02d}")
            cost_k = self.effective_cost_at_k()
            individual_costs_k = self.individual_costs_at_k()
            self.cost.append({"effective_cost": cost_k, **individual_costs_k}) # list of dicts, each dict for 1 time slot k
            self.placement.append({v: i for v in self.virtual_nodes for i in self.physical_nodes if self.phi_node.X[self.virtual_nodes_index[v], self.physical_nodes_index[i]] == 1})
            self.migrations.append(int(self.xi.X.sum())) # number of migrations for this time slot k
            self.overall_cost += cost_k
            print(f"Cost for time slot k={k}: {cost_k}")
            print(f"{int(self.gpmodel.NodeCount)} nodes explored in the branch and bound tree")
            print(f"{int(self.gpmodel.SolCount)} feasible solutions found")
            print(f"MIP gap: {self.gpmodel.MIPGap} \n")
        print(f"Cost for each time slot k: {self.cost}")
        print(f"Overall cost for the whole time horizon: {self.overall_cost}")

    def run_nomig(self):
        """
            Runs the model without migration, only optimizing for the current time slot k=0. \n
            This is triggered when W=0, to compare the results with migration so we can conclude on the benefits of VNF migration. \n
        """
        self.gpmodel.Params.OutputFlag = int(self.verbose)
        self.compute_model()
        self.optimize()
        cost_k = self.effective_cost_at_k()
        individual_costs_k = self.individual_costs_at_k()
        self.cost.append({"effective_cost": cost_k, **individual_costs_k})
        self.overall_cost += cost_k
        for k in range(1, min(len(self.energy_price[self.physical_nodes[0]]), self.N)):
            self.k += 1
            cost_k = self.effective_cost_at_k()
            individual_costs_k = self.individual_costs_at_k()
            self.cost.append({"effective_cost": cost_k, **individual_costs_k})
            self.overall_cost += cost_k
        #print(f"Cost for each time slot k: {self.cost}")
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
            Saves the plot as  `./plots/graphs/physical_graph.svg`\n

            Argument : graph_name (str) : name of the graph and the file to save, default is "physical_graph"
        """
        assert self.optimized_flag, "The model must have been optimized in order to generate the graph"
        plots_dir = Path(__file__).resolve().parent / self.graphviz_output_dir
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

        g.attr(label=f"k={self.k:02d}", labelloc="t", labeljust="l", fontsize="14", fontcolor="black")
        g.render(directory=str(plots_dir), cleanup=True)



if __name__ == "__main__":
    #model = json_parser("model3.2")
    model = json_parser("nobel-eu")
    network_mapping = NetworkMapping(model)
    network_mapping.run()
    #print("Physical Links:", network_mapping.physical_links)
    #print("Logical links: ", network_mapping.logical_links)
    # network_mapping.plot_all_graphs(k_values=range(len(network_mapping.energy_price[network_mapping.physical_nodes[0]])), n_cols=5)
