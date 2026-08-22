""" ABOUT THIS FILE.
    This file contains the class NetworkMapping, adapted from optim_on_whole_window.py to optimize the mapping of VNFs to physical servers and the routing of logical links to physical links over the whole observation window, but with the integrality constraint relaxed on every time step of the window except the first one (w=0).
    Only the decision that is effectively applied (w=0) has to be an actual placement, the remaining time steps of the window are just here to give the optimizer a foresight of the future costs. Relaxing them to continuous variables in [0,1] gives a worse estimation of the true cost, but a sufficient one, and the optimization is way faster : we lose a bit of optimality to reduce the complexity of the optimization.
    Two formulations had to be adapted so that the relaxed part of the model stays linear (a product of two relaxed variables would be a nonconvex bilinear term, which would defeat the purpose of the relaxation) :
    - the migration cost on the relaxed time steps uses a departure variable xi_out (see `generate_variables` and `generate_migration_constraints`), which is strictly equivalent to the original product formulation as long as the variables are binary,
    - the delay constraint only keeps its linear part (the processing delay of the hosted VNFs is dropped on every time step).
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

    def __init__(self, model: dict, W: int = 1, S: int = 1, N: int = 10, time_stride: int =1, offset: int = 0, prices_csv: str = "energy_prices.csv", graphviz_output_dir: str = "plots/graphs", time_limit: float = 30):
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

        self.static = (W<=0)    # W=0 is the oracle static, W=-1 the myopic one
        self.oracle = (W==0)
        if self.static:
            self.W = 1

        # Gurobi model hyperparameters
        self.gpmodel = gp.Model("mip1")
        #self.gpmodel.Params.MIPGap = 1e-9          # too restrictive
        #self.gpmodel.Params.MIPGapAbs = 1e-12      # too restrictive
        self.gpmodel.Params.TimeLimit = time_limit  # safety net : optimize() always returns within this budget, best solution found so far is kept either way
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
        self.placement = []
        self.migrations = []
        self.mip_gaps = []      # gurobi MIP gap of each time slot, to check afterwards that the results are not just a truncated branch and bound
        self.statuses = []      # gurobi status of each time slot, same purpose
        self.time_stride = time_stride

        self.graphviz_output_dir = graphviz_output_dir
        self.verbose = False

        self.max_delay = 0.15 # s, 150ms SLA per SFC now
        #self.max_delay = model["max_delay"]
        self.migration_downtime = 0.228 # in seconds. Comes from Liu2011 (Dbench benchmark, 124ms scaled to a 2GB VNF) : closest workload to a NAT/FW/TM VNF apparently (otherwise it's incomparable)

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
        """
            generates the mapping variables for the VNFs to physical servers and for the logical links to physical links \n
            Every variable is declared as binary on the whole window and then relaxed to a continuous variable in [0,1] for w>=1, so that the integrality is only enforced on the time step we actually apply (w=0). Relaxing the type of a binary variable keeps its bounds [0,1], which is exactly the relaxation we want.
        """
        # Numpy array of binary variables for the mapping of VNFs to physical servers (the only ones we need for now)
        self.phi_node = self.gpmodel.addMVar((self.W, len(self.virtual_nodes), len(self.physical_nodes)), vtype=GRB.BINARY, name="phi_nodes")
        self.phi_link = self.gpmodel.addMVar((self.W, len(self.logical_links), len(self.physical_links)), vtype=GRB.BINARY, name="phi_link")
        # node activation variables, sigma_i = 1 if at least one VNF is mapped to node i, 0 otherwise
        self.sigma = self.gpmodel.addMVar((self.W, len(self.physical_nodes)), vtype=GRB.BINARY, name="sigma")
        # migration variables, xi_v,i = 1 if VNF v is migrated to node i, 0 otherwise
        self.xi = self.gpmodel.addMVar((self.W, len(self.virtual_nodes), len(self.physical_nodes)), vtype=GRB.BINARY, name="xi")
        # departure variables, xi_out_v,i = 1 if VNF v leaves node i, 0 otherwise. those vars only exist to keep the migration cost linear on the relaxed time steps, the first step w=0 doesn't need them cause the previous placement is a constant
        self.xi_out = self.gpmodel.addMVar((self.W, len(self.virtual_nodes), len(self.physical_nodes)), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="xi_out")
        self.xi_out[0, :, :].UB = 0

        # INTEGER RELAXATION : every time step of the window except for the first one is relaxed
        if self.W > 1:
            self.phi_node[1:, :, :].VType = GRB.CONTINUOUS
            self.phi_link[1:, :, :].VType = GRB.CONTINUOUS
            self.sigma[1:, :].VType = GRB.CONTINUOUS
            self.xi[1:, :, :].VType = GRB.CONTINUOUS
        self.gpmodel.update()    # needed because we modified attributes of the variables and not only added them

    def generate_mapping_constraints(self):
        """ generates the mapping constraints on phi_node and phi_link """

        for k in range(self.W):
            # Each VNF must be mapped to exactly one physical server
            for v in range(len(self.virtual_nodes)):
                self.gpmodel.addConstr(
                    gp.quicksum(self.phi_node[k, v, i] for i in range(len(self.physical_nodes))) == 1
                )
            # Flow conservation constraints for the logical links
            for i_index, i in enumerate(self.physical_nodes):
                for vlink_index, (v, w) in enumerate(self.logical_links):
                    self.gpmodel.addConstr(
                        gp.quicksum(
                            self.phi_link[k, vlink_index, self.physical_link_index[(i, j)]]
                            - self.phi_link[k, vlink_index, self.physical_link_index[(j, i)]]
                            for j_index, j in enumerate(self.physGraph[i])
                        )
                        == self.phi_node[k, self.virtual_nodes_index[v], i_index] - self.phi_node[k, self.virtual_nodes_index[w], i_index]
                    )

    def generate_node_activation_constraints(self):
        # sigma_i = 1 if at least one VNF is mapped to node i, 0 otherwise
        # On forums they say that this way is better than using a GenConstr because this form is how Gorubi enventually translates the constraint into
        # this formulation is also the reason why the relaxed sigma still behaves well : it is pushed down to the largest phi_node of the node by the minimization
        for k in range(self.W):
            for i_index in range(len(self.physical_nodes)):
                for v in range(len(self.virtual_nodes)):
                    self.gpmodel.addConstr(self.sigma[k, i_index] >= self.phi_node[k, v, i_index])
                self.gpmodel.addConstr(self.sigma[k, i_index] <= gp.quicksum(self.phi_node[k, v, i_index] for v in range(len(self.virtual_nodes)))) # needed when the price is negative : otherwise switching empty servers on would earn money
                self.gpmodel.addConstr(self.sigma[k, i_index] >= gp.quicksum(self.phi_node[k, v, i_index] * self.computing_requirements[self.virtual_nodes[v]] for v in range(len(self.virtual_nodes))) / self.computing_availability[self.physical_nodes[i_index]]) # valid for a binary placement, and it stops a fractionally filled node from paying only a fraction of its idle power on the relaxed time steps

    def generate_availability_constraints(self):

        for k in range(self.W):
            # Availability constraints, applied to every physical node since access nodes can host VNFs too
            for i_index, i in enumerate(self.physical_nodes):
                # In terms of computing resource
                self.gpmodel.addConstr(
                    gp.quicksum(
                        self.phi_node[k, v_index, i_index] * self.computing_requirements[v]
                        for v_index, v in enumerate(self.virtual_nodes)
                    ) <=  self.computing_availability[self.physical_nodes[i_index]]
                )
                # In terms of memory resource
                self.gpmodel.addConstr(
                    gp.quicksum(
                        self.phi_node[k, v_index, i_index] * self.memory_requirements[v]
                            for v_index, v in enumerate(self.virtual_nodes)
                    ) <=  self.memory_availability[self.physical_nodes[i_index]]
                )

            # And in terms of bandwidth usage
            for i, j in self.physical_links:
                self.gpmodel.addConstr(
                    gp.quicksum(
                        self.phi_link[k, vlink_index, self.physical_link_index[(i, j)]] * self.bandwidth_requirement[vlink_index]
                            for vlink_index, (v,w) in enumerate(self.logical_links)
                    ) <= self.bandwidth_availability[self.physical_link_index[(i, j)]]
                )

    def generate_access_nodes_constraints(self):
        for k in range(self.W):
            # First VNF must be mapped to the first access node and the last VNF must be mapped to the last access node
            for access_node in self.access_nodes.items():
                # First VNF must be mapped to the first access node and the last VNF must be mapped to the last access node
                self.gpmodel.addConstr(
                    self.phi_node[
                        k,
                        self.virtual_nodes_index[access_node[0]],
                        self.physical_nodes_index[access_node[1]]
                    ] == 1
                )

    def generate_migration_constraints(self):
        """
            Intra-window migration constraints, generated only once at the beginning of the optimization\n
            xi_out is constrained the same way as xi but with the two time steps swapped, so that it tells us which node a VNF leaves instead of which node it joins. 
        """
        self.migration_constrs = []
        for w in range(1, self.W): # i changed the variable name from k to w to avoid confusion with the time slot k (relative vs absolute) but not everywhere
            for v in range(len(self.virtual_nodes)):
                for i in range(len(self.physical_nodes)):
                    self.migration_constrs += [
                        self.gpmodel.addConstr(self.xi[w,v,i] >= self.phi_node[w,v,i] - self.phi_node[w-1,v,i]),
                        self.gpmodel.addConstr(self.xi[w,v,i] <= self.phi_node[w,v,i]),
                        self.gpmodel.addConstr(self.xi[w,v,i] <= 1 - self.phi_node[w-1,v,i]),
                        self.gpmodel.addConstr(self.xi_out[w,v,i] >= self.phi_node[w-1,v,i] - self.phi_node[w,v,i]),
                        self.gpmodel.addConstr(self.xi_out[w,v,i] <= self.phi_node[w-1,v,i]),
                        self.gpmodel.addConstr(self.xi_out[w,v,i] <= 1 - self.phi_node[w,v,i]),
                    ]

    def refresh_boundary_migration_constraints(self):
        """contraint for the link w=0 with the previous placement. It'll refresh this at each step."""
        if getattr(self, "boundary_constrs", None):
            self.gpmodel.remove(self.boundary_constrs)
        self.boundary_constrs = []
        if self.k == 0: # no migration on the first time slot, since it's just the beginning of the lifecycle
            for v in range(len(self.virtual_nodes)):
                for i in range(len(self.physical_nodes)):
                    self.boundary_constrs.append(self.gpmodel.addConstr(self.xi[0,v,i] == 0))
        else:
            prev = np.rint(self.prev_phi_node[0])     # for some reason, gurobi returns a float so we need to round it
            for v in range(len(self.virtual_nodes)):
                for i in range(len(self.physical_nodes)):
                    p = float(prev[v, i])
                    self.boundary_constrs += [
                        self.gpmodel.addConstr(self.xi[0,v,i] >= self.phi_node[0,v,i] - p),
                        self.gpmodel.addConstr(self.xi[0,v,i] <= self.phi_node[0,v,i]),
                        self.gpmodel.addConstr(self.xi[0,v,i] <= 1 - p),
                    ]
        self.gpmodel.update()

    def sfc_partition(self):
        """
            Splits the virtual graph into its individual SFCs, taken as the weakly connected components of the
            logical links. Deriving them from the structure of the graph rather than from the VNF names keeps this
            working for every naming convention I used so far ("v1"/"v1'", "s1v1", "v1_1"...).

            Returns: list of dicts {"nodes": [indices in self.virtual_nodes], "links": [indices in self.logical_links]}
        """
        if getattr(self, "_sfc_partition", None) is not None:
            return self._sfc_partition

        parent = {v: v for v in self.virtual_nodes}     # union-find, the cheapest way to get the connected components
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for v, w in self.logical_links:
            root_v, root_w = find(v), find(w)
            if root_v != root_w:
                parent[root_v] = root_w

        groups = {}
        for v in self.virtual_nodes:
            groups.setdefault(find(v), {"nodes": [], "links": []})["nodes"].append(self.virtual_nodes_index[v])
        for link_index, (v, w) in enumerate(self.logical_links):
            groups[find(v)]["links"].append(link_index)

        self._sfc_partition = list(groups.values())
        return self._sfc_partition


    def generate_delay_constraints(self):
        """
            Generates the constraints on the total delay introduced by the routing of the SFCs onto the physical network.\n
            The formula that I'll use is : $\sum_{vw\in logical_links of s} \sum_{ij\in physical_links} phi_link_{vw:ij} * d_{ij} <= max_delay  forall SFC s$ \n
            One constraint is generated per SFC and per time step, so max_delay is a per-chain SLA 
            If this constraint isn't verified, the optimizer will cancel the mapping, so TODO make it non destructive

            Any estimation of the delay required for the transmission of a packet based on the distance between nodes will be rough, because it depends on the actual network topology and traffic conditions. We may consider a reasonable and usual value of a 0.5ms delay per 100km of otpical fiber + 50us of switching per hop

            The processing delay of the VNFs is dropped on every time step because it is a product of two variables (phi_link * phi_node), so a quadratic constraint
        """
        delay_per_100km = 0.0005 # 0.5ms per 100km of optical fiber
        delay_per_hop = 0.00005 # 50us per hop (switching delay)
        delay_per_VNF = 0.002 # 2ms per VNF hosting (processing delay, idk if it's realistic)

        for sfc in self.sfc_partition():
            # the first time step keeps the exact formulation, since phi_node[0] and phi_link[0] are still binary
            self.gpmodel.addConstr(
                gp.quicksum(
                    self.phi_link[0, vw_index, ij_index] * (
                        self.links_distance_dict[ij[0]][ij[1]] * delay_per_100km
                        + delay_per_hop # corresponds to the switching delay at the physical nodes
                        # + self.phi_node[0, self.virtual_nodes_index[self.logical_links[vw_index][0]], self.physical_nodes_index[ij[0]]] * delay_per_VNF # quadratic constraint
                    )
                    for ij_index, ij in enumerate(self.physical_links)
                    for vw_index in sfc["links"]
                )
                <= self.max_delay
            )
            for k in range(1, self.W): # relaxed time steps : linear part of the delay only, otherwise too long
                self.gpmodel.addConstr(
                    gp.quicksum(
                        self.phi_link[k, vw_index, ij_index] * (
                            self.links_distance_dict[ij[0]][ij[1]] * delay_per_100km
                            + delay_per_hop # corresponds to the switching delay at the physical nodes
                        )
                        for ij_index, ij in enumerate(self.physical_links)
                        for vw_index in sfc["links"]
                    )
                    <= self.max_delay
                )


    def energy_cost(self, k=None, w=None):
        """
            REALISTIC ENERGY COST FUNCTION :\n
            From the real energy price in Europe, the current CPU load, and considering the servers to be the SPECpower reference Dell PowerEdge HS5610, we can estimate the energy cost for each physical server at each time slot k: \n
            $C_e(k) = \sum_{i\in physical_nodes} energy_price_i(k) * P_i(k)$ \n
            with:\n
            $P_i(k) = P_idle + (P_max - P_idle) * CPU_usage_i(k)$ \n
            $CPU_usage_i(k) = \sum_{v\in VNFs} phi_node_{v,i}(k) * computing_requirement_v / computing_availability_i$ \n
            This cost is linear in phi_node and sigma, so it doesn't need to be adapted for the relaxed time steps.
        """
        # we used to depend on a parameter from the json but now I use the Dell PowerEdge HS5610 SPECpower specifications and CPU usage
        # depends on k
        if k is None:
            k = self.k
        if w is None:
            w = k - self.k
        # self.Ce = gp.quicksum(
        #     self.energy_price[self.physical_nodes[i]][k] * gp.quicksum(
        #         self.phi_node[v, i] for v in range(len(self.virtual_nodes))
        #     ) for i in range(len(self.physical_nodes))
        # )
        Watts_over_DeltaT_to_MWh = 1/1000000 / 3600 * 900 * self.time_stride # 900 seconds = 15 minutes
        P_idle = 122 # Watts
        P_max = 602 # Watts
        CPU_usage = lambda i: gp.quicksum(
            self.phi_node[w, v, i] * self.computing_requirements[self.virtual_nodes[v]]
            for v in range(len(self.virtual_nodes))
        ) / self.computing_availability[self.physical_nodes[i]]
        scaling_Power = lambda i: (P_max - P_idle) * CPU_usage(i)
        fix_Power = lambda i: self.sigma[w, i] * P_idle
        self.Ce = gp.quicksum(
            self.energy_price[self.physical_nodes[i]][k] * Watts_over_DeltaT_to_MWh * (fix_Power(i) + scaling_Power(i))
            for i in range(len(self.physical_nodes))
        )
        return self.Ce

    def disposal_cost(self, k=None, w=None): # à revoir
        if k is None:
            k = self.k # but unused fot the moment
        if w is None:
            w = k - self.k
        self.Cf = gp.quicksum(
            self.node_disposal_price[self.physical_nodes[i]] * self.sigma[w, i] for i in range(len(self.physical_nodes))
        )
        return self.Cf

    def usage_cost(self, k=None, w=None):   # à revoir
        if k is None:
            k = self.k
        if w is None:
            w = k - self.k
        self.Cr = gp.quicksum(
            gp.quicksum(
                self.CPU_usage_price[self.physical_nodes[i]] * self.phi_node[w, v, i] * self.computing_requirements[self.virtual_nodes[v]]
                for v in range(len(self.virtual_nodes))
            )
            + gp.quicksum(
                self.memory_usage_price[self.physical_nodes[i]] * self.phi_node[w, v, i] * self.memory_requirements[self.virtual_nodes[v]]
                for v in range(len(self.virtual_nodes))
            )
            # + gp.quicksum(
            #     self.bandwidth_usage_price[self.physical_nodes[i]][self.physical_nodes[j]]
            #     * self.phi_link[k - self.k, v_link, self.physical_link_index[(self.physical_nodes[i], self.physical_nodes[j])]]
            #     * self.bandwidth_requirement[v_link]
            #     for v_link in range(len(self.logical_links))
            #     for j in range(len(self.physical_nodes)) if (self.physical_nodes[i], self.physical_nodes[j]) in self.physical_links
            # ) # no cost is associated with bandwidth usage
            for i in range(len(self.physical_nodes))
        )
        return self.Cr

    def link_delay(self, k=None, w=None): #
        if k is None:
            k = self.k
        if w is None:
            w = k - self.k
        delay_per_100km = 0.0005 # 0.5ms per 100km of optical fiber
        delay_per_hop = 0.00005 # 50us per hop (switching delay)
        delay_per_VNF = 0.002 # 2ms per VNF hosting (processing delay, idk if it's realistic)
        self.total_link_delay = gp.quicksum(
            self.phi_link[w, vw_index, ij_index] * (
                self.links_distance_dict[ij[0]][ij[1]] * delay_per_100km
                + delay_per_hop # corresponds to the switching delay at the physical nodes
                # + self.phi_node[w, self.virtual_nodes_index[vw[0]], self.physical_nodes_index[ij[0]]] * delay_per_VNF # quadratic constraint
            )
            for ij_index, ij in enumerate(self.physical_links)
            for vw_index, vw in enumerate(self.logical_links)
        )
        return self.total_link_delay

    def link_delay_cost(self, k=None, w=None, coefficient=0.001):
        """
           This was our fictious cost to reduce delay. It is disabled : the function returns 0 so that nothing but the energy and the migration appears in the objective
           It is supposed to be negligible compared to the other costs, but it will be used to make the optimizer choose the shortest paths for the logical links, and thus simplify the final graph (many path are available). \n
        """
        return gp.LinExpr(0) # so negligible that it won't be taken into account in the optimization, but still takes time to compute
        if k is None:
            k = self.k
        self.Cl = coefficient * self.link_delay(k=k, w=w)
        return self.Cl

    def migration_cost(self, k=None, coefficient=1):
        """
            Same cost as in the exact model, but written so that it stays linear on the relaxed time steps. \n
            On the first step (w=0) the previous placement is a constant, so xi * previous_placement is already linear so no need to modify. \n
            On the relaxed steps the origin would be a variable too, so instead of the product xi * phi_node we charge the origin server through the departure variable xi_out : since a VNF leaves at most one node and joins at most one node, $E_origin * price_i * xi_out_{v,i} + E_dest * price_j * xi_{v,j}$ gives exactly the same cost as the product formulation when the variables are binary.\n
            That's how we can keep the migration cost linear even on the relaxed time steps
        """
        if k is None:
            k = self.k
        Joules_to_MWh = 1 / 1000000 / 3600
        P_idle = 122                         # W
        fix_migration_energy = 663          # Joules, Liu2011 (Dbench, 1326J total for a 2GB VNF, split per server)
        migration_duration = 42.6        # seconds, Liu2011 (Dbench migration latency scaled to a 2GB VNF)

        E_origin = fix_migration_energy + P_idle * migration_duration   # origin server stays on
        E_dest   = fix_migration_energy                                 # P_idle for destination server is already in energy_cost

        if k == self.k:
            if self.k == 0:
                return gp.LinExpr(0)                    # no migration cost for the first time slot
            w = 0
            prev = np.rint(self.prev_phi_node[0])
            origin_price = {v: sum(self.energy_price[i][k] * float(prev[self.virtual_nodes_index[v], self.physical_nodes_index[i]]) for i in self.physical_nodes) for v in self.virtual_nodes}
            self.Cm = gp.quicksum(
                coefficient * Joules_to_MWh
                * (E_origin * origin_price[v] + E_dest * self.energy_price[j][k])
                * self.xi[w, self.virtual_nodes_index[v], self.physical_nodes_index[j]]
                for v in self.virtual_nodes
                for j in self.physical_nodes
            )
        else:
            w = k - self.k
            self.Cm = gp.quicksum(
                coefficient * Joules_to_MWh * E_origin * self.energy_price[i][k]
                * self.xi_out[w, self.virtual_nodes_index[v], self.physical_nodes_index[i]]
                for v in self.virtual_nodes
                for i in self.physical_nodes # for the origin node
            ) + gp.quicksum(
                coefficient * Joules_to_MWh * E_dest * self.energy_price[j][k]
                * self.xi[w, self.virtual_nodes_index[v], self.physical_nodes_index[j]]
                for v in self.virtual_nodes
                for j in self.physical_nodes # for the destination node
            )
        return self.Cm

    def total_window_cost(self):
        """
            Total cost over the time window W, for the foresighted model. \n
            This function will be used to compute the total cost over the time window W, and will be used in the objective function\n
            Only the w=0 part of it is an exact cost, the rest is the relaxed estimation of the future costs as can see in the previous docstirngs
        """

        if self.static: # just one time slot, no migration cost
            self.total_cost = gp.quicksum(self.energy_cost(k, w=0) + self.link_delay_cost(k, w=0) for k in (range(self.N) if self.oracle else range(1))) # the myopic static only sees the first time slot
            return self.total_cost

        self.total_cost = gp.quicksum(
            self.energy_cost(k)  + self.link_delay_cost(k) + self.migration_cost(k)
            # + self.usage_cost() + self.disposal_cost(k)
            for k in range(self.k, self.k + self.W)
        )
        return self.total_cost

    def total_window_objective_function(self):
        self.gpmodel.setObjective(self.total_window_cost(), GRB.MINIMIZE)

    def effective_cost_at_k(self):
        """
            Cost at time slot k, used to save the cost at each time slot and display the cumulative cost at the end of the optimization process. \n
            This one is not affected by the relaxation since it only involves the variables of the applied time step, which are still binary
        """
        if self.static: # no migration cost
            return self.energy_cost().getValue()

        # cost_k = self.energy_cost() + self.usage_cost() + self.disposal_cost() + self.migration_cost()
        cost_k = self.energy_cost() + self.migration_cost()
        return cost_k.getValue()

    def individual_costs_at_k(self):
        """
            Individual effective costs at time slot k, for energy, migration, usage, and disposal, + the delay
        """
        if self.static: # no migration cost
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
        self.generate_migration_constraints()
        self.refresh_boundary_migration_constraints()
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

        self.refresh_boundary_migration_constraints()
        self.total_window_objective_function()
        self.gpmodel.update()


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
            self.statuses.append(int(self.gpmodel.Status))
            self.mip_gaps.append(float(self.gpmodel.MIPGap) if self.gpmodel.SolCount > 0 else float('inf'))
            if self.gpmodel.Status == GRB.OPTIMAL:
                if self.verbose:
                    for v in self.gpmodel.getVars():
                        print(f"{v.VarName} {v.X:g}")
                print(f"Obj: {self.gpmodel.ObjVal:g}")
                self.optimized_flag = 1
                # self.plot_graph()
            elif self.gpmodel.Status == GRB.TIME_LIMIT:
                if self.gpmodel.SolCount > 0:
                    print(f"Time limit reached ({self.gpmodel.Params.TimeLimit:g}s) : optimality not proven, "
                        + f"so we're using the best solution found (gap={self.gpmodel.MIPGap*100:.2f}%, obj={self.gpmodel.ObjVal:g}, bound={self.gpmodel.ObjBound:g})")
                    self.optimized_flag = 1
                else:
                    raise ValueError(f"Time limit reached ({self.gpmodel.Params.TimeLimit:g}s) without finding any feasible solution") # this shouldn't happen normally but who knows
            elif self.gpmodel.Status == GRB.INFEASIBLE:
                raise ValueError(f"Model is infeasible at k={self.k} (W={self.W})")
            else:
                raise ValueError(f"Optimization finished with status {self.gpmodel.Status} at k={self.k} (W={self.W})")

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")
        except AttributeError:
            print("Encountered an attribute error")

    def run(self):
        """
            Runs the model by computing it and optimizing it for each time slot k.
        """
        if self.static:
            print(f"W<=0 => running without migration, {'oracle' if self.oracle else 'myopic'} static placement")
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
        self.placement.append({v: i for v in self.virtual_nodes for i in self.physical_nodes if self.phi_node.X[0, self.virtual_nodes_index[v], self.physical_nodes_index[i]] == 1})
        self.migrations.append(0) # no migration at k=0
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
            self.placement.append({v: i for v in self.virtual_nodes for i in self.physical_nodes if self.phi_node.X[0, self.virtual_nodes_index[v], self.physical_nodes_index[i]] == 1})
            self.migrations.append(int(np.rint(self.xi.X[0]).sum())) # number of migrations for this time slot k, only w=0 is counted since the relaxed time steps are fractional and are not applied
            self.overall_cost += cost_k
            print(f"Cost for time slot k={k}: {cost_k}")
            print(f"{int(self.gpmodel.NodeCount)} nodes explored in the branch and bound tree")
            print(f"{int(self.gpmodel.SolCount)} feasible solutions found")
            print(f"MIP gap: {self.gpmodel.MIPGap} \n")
        #print(f"Cost for each time slot k: {self.cost}")
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
            Saves the plot as  `./plots/graphs/physical_graph.svg`\n
            Only the applied time step (w=0) is plotted, so the relaxation doesn't affect this method

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
            i_used  =   [self.virtual_nodes[v] for v in range(len(self.virtual_nodes)) if self.phi_node.X[0, v, self.physical_nodes_index[i]] == 1]
            j_used  =   [self.virtual_nodes[v] for v in range(len(self.virtual_nodes)) if self.phi_node.X[0, v, self.physical_nodes_index[j]] == 1]
            ij_used =   [self.logical_links[v_link] for v_link in range(len(self.logical_links)) if self.phi_link.X[0, v_link, self.physical_link_index[(i, j)]] == 1]

            link_label = f"{self.phi_link.X[0, :, self.physical_link_index[(i, j)]] @ self.bandwidth_requirement[:]}/{self.bandwidth_availability[self.physical_link_index[(i, j)]]}"

            node_i_compute = f" \n c: {sum(self.phi_node.X[0, index_v, self.physical_nodes_index[i]] * self.computing_requirements[v] for index_v, v in enumerate(self.virtual_nodes))}/{self.computing_availability[i]}"
            node_i_memory = f" \n m: {sum(self.phi_node.X[0, index_v, self.physical_nodes_index[i]] * self.memory_requirements[v] for index_v, v in enumerate(self.virtual_nodes))}/{self.memory_availability[i]}"
            node_i_energy_price = f" \n e: {self.energy_price[i][self.k]}"
            node_i_label = f"{i} ({i_used})" + node_i_compute + node_i_memory + node_i_energy_price

            node_j_compute = f" \n c: {sum(self.phi_node.X[0, index_v, self.physical_nodes_index[j]] * self.computing_requirements[v] for index_v, v in enumerate(self.virtual_nodes))}/{self.computing_availability[j]}"
            node_j_memory = f" \n m: {sum(self.phi_node.X[0, index_v, self.physical_nodes_index[j]] * self.memory_requirements[v] for index_v, v in enumerate(self.virtual_nodes))}/{self.memory_availability[j]}"
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
