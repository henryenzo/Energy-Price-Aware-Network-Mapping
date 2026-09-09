"""
    Generator of nobel-eu test-model entries parameterised by the network load factor rho.

    Each generated SFC has sfc_len VNFs (8 by default) whose first and last ones are
    fictitious: they are the access nodes of the chain, they are pinned to a physical node
    and they carry no CPU/memory load. Only the sfc_len - 2 inner VNFs count in rho.

    VNFs are named "v<position>_<instance>", e.g. "v2_3" is the 2nd VNF of the 3rd SFC.

    Created on August 6th, 2026
"""

from data_engine import *

import os


# order of the keys to write, the empty strings are the blank lines of test_models.json
_KEY_LAYOUT = [
    "physGraph", "virtualGraph", "access_nodes", "",
    "computing_availability", "memory_availability", "bandwidth_availability_dict", "",
    "computing_requirements", "memory_requirements", "bandwidth_requirements_dict", "",
    "energy_price", "",
    "CPU_usage_price", "memory_usage_price", "bandwidth_usage_price",
    "node_disposal_price", "node_country", "",
    "migration_energy_cost", "links_distance_dict",
]


def _format_entry(model_name: str, description: str, params: dict) -> str:
    """
        Serialises a model entry with exactly the layout used in test_models.json:
        one line per model_parameters key, dicts dumped inline.
    """
    keys = [k for k in _KEY_LAYOUT if k == "" or k in params]
    while keys and keys[-1] == "":
        keys.pop()

    lines = []
    last_real = max(i for i, k in enumerate(keys) if k != "")
    for i, key in enumerate(keys):
        if key == "":
            lines.append("")
            continue
        comma = "" if i == last_real else ","
        lines.append(f'                "{key}": {json.dumps(params[key])}{comma}')

    body = "\n".join(lines)
    return (
        "        {\n"
        f'            "model_name": "{model_name}",\n'
        f'            "description": "{description}",\n'
        '            "model_parameters": {\n'
        f"{body}\n"
        "            }\n"
        "        }"
    )


def generate_nobel_eu_entry(n_sfc: int,
                            cpu_req=4, mem_req=2, bw_req=2,
                            cpu_avail=64, mem_avail=256, bw_avail=100,
                            sfc_len=8, dummy_req=0, migration_cost=1,
                            model_name=None, description=None,
                            source_model="nobel-eu", source_file="test_models.json",
                            seed=None) -> str:
    """
        Generates a nobel-eu test-model entry with `n_sfc` instances of a `sfc_len`-VNF SFC.

        Arguments:
        - n_sfc:        int - number of SFC instances to generate
        - cpu_req:      int - CPU requirement of each real VNF
        - mem_req:      int - memory requirement of each real VNF
        - bw_req:       int - bandwidth requirement of each virtual link
        - cpu_avail:    int - CPU availability of every physical node
        - mem_avail:    int - memory availability of every physical node
        - bw_avail:     int - bandwidth availability of every physical link
        - sfc_len:      int - VNFs per SFC, the first and the last one being fictitious
        - dummy_req:    int - CPU/memory requirement of the two fictitious VNFs
        - migration_cost: int - migration_energy_cost of every VNF
        - model_name:   str - defaults to "nobel-eu-<n_sfc>SFC"
        - description:  str - defaults to a generated one
        - source_model: str - model whose topology (physGraph, prices, distances...) is reused
        - source_file:  str - json file holding the source model
        - seed:         int - seed for the random draw of the access nodes

        Returns:
        str - the formatted json entry, ready to be passed to append_entry_to_json()
    """
    if sfc_len < 3:
        raise ValueError("sfc_len must be >= 3 so that at least one real VNF remains")

    src = json_parser(source_model, source_file)
    nodes = list(src["computing_availability"].keys())
    if seed is not None:
        np.random.seed(seed)

    virtual_graph, access_nodes = {}, {}
    computing_req, memory_req, bandwidth_req, migration_req = {}, {}, {}, {}

    for k in range(1, n_sfc + 1):
        src_node, dst_node = np.random.choice(nodes, size=2, replace=False)
        access_nodes[f"v1_{k}"] = str(src_node)
        access_nodes[f"v{sfc_len}_{k}"] = str(dst_node)

        for j in range(1, sfc_len + 1):
            vnf = f"v{j}_{k}"
            succ = [f"v{j + 1}_{k}"] if j < sfc_len else []
            is_dummy = (j == 1 or j == sfc_len)

            virtual_graph[vnf] = succ
            computing_req[vnf] = dummy_req if is_dummy else cpu_req
            memory_req[vnf] = dummy_req if is_dummy else mem_req
            bandwidth_req[vnf] = {s: bw_req for s in succ}
            migration_req[vnf] = migration_cost

    params = {
        "physGraph": src["physGraph"],
        "virtualGraph": virtual_graph,
        "access_nodes": access_nodes,
        "computing_availability": {i: cpu_avail for i in nodes},
        "memory_availability": {i: mem_avail for i in nodes},
        "bandwidth_availability_dict": {i: {j: bw_avail for j in nbrs}
                                        for i, nbrs in src["bandwidth_availability_dict"].items()},
        "computing_requirements": computing_req,
        "memory_requirements": memory_req,
        "bandwidth_requirements_dict": bandwidth_req,
        "energy_price": src["energy_price"],
        "CPU_usage_price": src["CPU_usage_price"],
        "memory_usage_price": src["memory_usage_price"],
        "bandwidth_usage_price": src["bandwidth_usage_price"],
        "node_disposal_price": src["node_disposal_price"],
        "node_country": src["node_country"],
        "migration_energy_cost": migration_req,
        "links_distance_dict": src["links_distance_dict"],
    }

    rho = compute_rho(n_sfc, cpu_req, mem_req, cpu_avail, mem_avail, sfc_len, len(nodes))[2]
    if model_name is None:
        model_name = f"nobel-eu-{n_sfc}SFC"
    if description is None:
        description = (f"{n_sfc} SFCs of {sfc_len - 2} real VNFs ({cpu_req} CPU / {mem_req} GB each), "
                       f"rho = {rho:.1%}, Nobel-EU topology (28 nodes, 41 european physical links)")

    return _format_entry(model_name, description, params)


def compute_rho(n_sfc, cpu_req=4, mem_req=2, cpu_avail=64, mem_avail=256,
                sfc_len=8, n_nodes=28):
    """
        Computes the load factor rho = total demand / total capacity, for CPU and memory.
        Only the sfc_len - 2 real VNFs of each SFC are counted.

        Returns: (rho_cpu, rho_mem, rho) where rho = max(rho_cpu, rho_mem) is the binding one
    """
    real = sfc_len - 2
    rho_cpu = n_sfc * real * cpu_req / (n_nodes * cpu_avail)
    rho_mem = n_sfc * real * mem_req / (n_nodes * mem_avail)
    return rho_cpu, rho_mem, max(rho_cpu, rho_mem)


def print_rho_table(cpu_req=4, mem_req=2, cpu_avail=64, mem_avail=256,
                    sfc_len=8, n_nodes=28, max_rho=1.0):
    """
        Prints the load factor reached by each number of SFC instances, stopping on the
        last one that does not exceed max_rho (100% by default).
    """
    real = sfc_len - 2
    print(f"{n_nodes} nodes x {cpu_avail} cores / {mem_avail} GB "
          f"= {n_nodes * cpu_avail} cores / {n_nodes * mem_avail} GB")
    print(f"1 SFC = {real} real VNFs x {cpu_req} cores / {mem_req} GB "
          f"= {real * cpu_req} cores / {real * mem_req} GB\n")
    print(f"{'#SFC':>6} | {'#VNF':>6} | {'rho_cpu':>9} | {'rho_mem':>9} | {'rho':>9} | binding")
    print("-" * 62)

    n = 1
    while True:
        rho_cpu, rho_mem, rho = compute_rho(n, cpu_req, mem_req, cpu_avail, mem_avail,
                                            sfc_len, n_nodes)
        if rho > max_rho:
            break
        binding = "CPU" if rho_cpu >= rho_mem else "memory"
        print(f"{n:>6} | {n * real:>6} | {rho_cpu:>8.1%} | {rho_mem:>8.1%} | {rho:>8.1%} | {binding}")
        n += 1

    print(f"\nlast feasible: {n - 1} SFC instances (rho = "
          f"{compute_rho(n - 1, cpu_req, mem_req, cpu_avail, mem_avail, sfc_len, n_nodes)[2]:.1%})")


def append_entry_to_json(entry: str, file_name="load_models.json"):
    """
        Appends an entry generated by generate_nobel_eu_entry() to `file_name`,
        creating the file with an empty "test_models" list if it does not exist yet.
    """
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write('{\n    "test_models": [\n    ]\n}\n')

    with open(file_name, "r") as f:
        text = f.read()

    model_name = entry.split('"model_name": "', 1)[1].split('"', 1)[0]
    if f'"model_name": "{model_name}"' in text:
        raise ValueError(f"Model '{model_name}' already exists in {file_name}")

    head, _, tail = text.rpartition("]")
    head = head.rstrip()
    head = head + "\n" if head.endswith("[") else head + ",\n"

    with open(file_name, "w") as f:
        f.write(head + entry + "\n    ]" + tail)

    print(f"'{model_name}' written to {file_name}")


if __name__ == "__main__":
    print_rho_table(cpu_req=4, mem_req=2, cpu_avail=64, mem_avail=256)

    # 74 SFCs is the last feasible number before rho exceeds 100%
    for n in [15, 30, 45, 60, 74]:
        append_entry_to_json(generate_nobel_eu_entry(n, seed=0), "load_models.json")
