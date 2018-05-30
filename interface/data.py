from config import *
import os
import numpy as np
from syconn.utils.datahandler import load_mapped_skeleton, load_pkl2obj

def list_fn(wd="./", cat="neurons"):
    """
        List tracing filenames in directory
    """
    return [f for f in os.listdir(os.path.join(wd,cat)) if f.endswith(".k.zip")]

def list_ids(wd="./", cat="neurons"):
    """
        List Ids of tracings in input directory
    """
    return [fn_to_id(f) for f in list_fn(wd, cat)]

def fn_to_id(fn):
    """
        Input : Filename (str)
        Output: id (int)
        Retrieve the tracing id from filename
    """
    return int(fn[:-6].split("_")[-1])

def id_to_fn(idx):
    """
        Input:  id (int)
        Output: Filename (str) 
        Creates filename from id
    """
    return "iter_0_{}.k.zip".format(idx)

def get_skel(idx, wd="./", cat="neurons"):
    """
        Returns the (mapped) skelton from either the id or filename of a tracing
        Inputs: idx
                wd
                cat
        output: Mapped Skelton
    """
    if isinstance(idx, str):
        fn = idx
    else:
        fn = id_to_fn(idx)
    return load_mapped_skeleton(os.path.join(wd, cat, fn), True, True)[0]

def count_nodes(x, key="axoness_pred"):
    """
        Input Skelton
        Output list[3], list[3]
        
        Output the number of nodes per axoness and spiness 
        for input skelton
    """
    assert key in ["axoness_pred", "spiness_pred"]
    counts = np.asarray([0,0,0])
    for node in list(x.nodes):
        idx = int(node.data[key])
        counts[idx] +=1
    return counts

def parse_subpart(node, tpe, key, nodes, edges):
    """
        Parse the signle subpart containing node
    """
    nnodes = list(edges[node])
    nnodes = filter(lambda x: not (x in nodes), nnodes)
    nnodes = filter(lambda x: int(x.data[key]) == tpe, nnodes)
    nodes  = nodes.union(set(nnodes))
    for node in nnodes:
        nodes = iter_parse(node, tpe, key, nodes, edges)
    return nodes

def parse_subparts(skel, key="axoness_pred"):
    """
        Parse a skelton into its subparts
    """
    parts = []
    edges = _get_bi_edges(skel)
    nodes = set(edges.keys())
    while len(nodes) > 0:
        curr_node  = nodes.pop()
        curr_type = int(curr_node.data[key])
        curr_nodes = parse_subpart(curr_node, curr_type, key, {curr_node}, edges)
        parts.append((curr_type, curr_nodes))
        nodes = nodes.difference(curr_nodes)
    return parts

def dist(skel, subset=None):
    """
        Computes the length of the subpart
    """
    if subset is None:
        subset = skel.nodes
    d = 0
    coords = {node: np.asarray(node.getCoordinate_scaled()) for node in subset}
    for node_start in subset:
        edges = [x for x in list(skel.edges[node_start]) if x in subset]
        for node_end in edges:
            d += np.linalg.norm(coords[node_start] - coords[node_end])
    return d

def dists(skel, key="axoness_pred"):
    """
        Computes the length of the subpart
    """
    counts = np.asarray([0,0,0])
    ds     = np.asarray([0,0,0])
    subparts = parse_subparts(skel, key)
    for tpe, part in subparts:
        counts[tpe]+=1
        ds[tpe]+= dist(skel, part)
    return counts, ds

def get_cell_type(idx, wd="./"):
    """
        DOC
    """
    return load_pkl2obj(wd + '/neurons/celltype_pred_dict.pkl')[idx]

def get_sj(wd):
    """
        DOC
    """
    mapidx = {v:k for k,v in load_pkl2obj(wd + 'contactsites/id_mapper.pkl').items()}
    mat    = np.load(wd + 'contactsites/syn_matrix.npy')
    return np.asarray([(mapidx[i], mapidx[j]) for i,j in zip(*np.where(mat))])

def merge_skels(skels):
    """
        DOC
    """
    skel = skels[0]
    for annos in skels[1:]:
        for node in annos.getNodes():
            skel.addNode(node)
        skel.edges.update(annos.edges)

    skel.hull_coords  = np.concatenate([skel.hull_coords for skel in skels], 0)
    skel.hull_normals = np.concatenate([skel.hull_normals for skel in skels], 0)
    skel.mito_hull_coords = np.concatenate([skel.mito_hull_coords for skel in skels], 0)
    skel.vc_hull_coords = np.concatenate([skel.vc_hull_coords for skel in skels], 0)
    skel.sj_hull_coords = np.concatenate([skel.sj_hull_coords for skel in skels], 0)
    return skel

def get_skels(ids):
    return merge_skels([get_skel(idx) for idx in ids])
    
def _get_bi_edges(skel):
    """
        DOC
    """
    edges  = skel.getEdges().copy()
    redges = skel.getReverseEdges()
    for edge in redges:
        x=redges[edge]
        edges[edge] = edges[edge].union(x)
    return edges
