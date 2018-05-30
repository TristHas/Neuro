from config import *
import numpy as np
import pandas as pd
from vispy.scene import visuals, SceneCanvas

from syconn.utils.illustration import *
from syconn.utils.datahandler  import *
from syconn.processing.cell_types import get_cell_type_labels
from syconn.utils import segmentationdataset# import SegmentationObject

from data import get_skel, get_skels

def split_hull(skel, subparts="axoness_pred"):
    """
    """
    kdtree = spatial.cKDTree([n.getCoordinate_scaled() for n in skel.nodes])
    nn_ixs = kdtree.query(skel.hull_coords, k=1)[1]

    skel_parts = [[],[],[]]
    for i,node in enumerate(skel.nodes):
        skel_parts[int(node.data[subparts])].append(skel.hull_coords[nn_ixs==i])

    for i in range(3):
        if skel_parts[i]:
            skel_parts[i] = np.concatenate(skel_parts[i])
            
    return skel_parts

def obj_coords(skel, center=np.asarray([0,0,0])):
    return (skel.mito_hull_coords - center, 
            skel.vc_hull_coords - center, 
            skel.sj_hull_coords - center)

def lines_from_skel(skel, center=np.asarray([0,0,0])):
    lines = [(k.getCoordinate_scaled()-center, w.getCoordinate_scaled()-center) for k,v in skel.edges.items() for w in list(v)]
    return [np.concatenate([np.expand_dims(l, 0) for l in line]) for line in lines]

def plot_hull(skel, subparts='spiness_pred', center=np.asarray([0,0,0])):
    assert subparts in [None, 'spiness_pred', 'axoness_pred']
    scatters = []
    if subparts:
        hulls  = [x-center if len(x) else x for x in split_hull(skel, subparts)]
        colors = [(1, 0.7, 0.7, .3), (0.7, 1, 0.7, .3), (0.7, 0.7, 1, .3)]
    else:
        hulls  = [skel.hull_coords - center]
        colors = [(1, 1, 1, .3)]
        
    for hull, color in zip(hulls, colors):
        if len(hull) > 0:
            scatter = visuals.Markers()
            scatter.set_data(hull, edge_color=None, face_color=color, size=5)
            scatters.append(scatter)
    return scatters

def plot_line(skel, center=np.asarray([0,0,0])):
    line_color = (1, 0, 0, 1)
    lines = []
    segments = lines_from_skel(skel, center)
    for l in segments:
        line = visuals.Line(l, color=line_color, width=10)
        lines.append(line)
    return lines

def plot_objects(skel, mi=True, vc=True, sj=True, center=np.asarray([0,0,0])):
    colors  = [(0, 0, 1, .5), (0, 1, 0, .5), (1, 0, 0, .5)]
    mi_vc_sj = obj_coords(skel, center)
    scatters = []
    for i in range(3):
        if [mi,vc,sj][i] and len(mi_vc_sj[i]) > 0:
            scatter = visuals.Markers()
            scatter.set_data(mi_vc_sj[i], edge_color=None, face_color=colors[i], size=5)
            scatters.append(scatter)
    return scatters
            
def plot(idx, line=True, subparts=None, mi=True, vc=True, sj=True, centering=True):
    skel   = get_skel(idx)
    canvas = SceneCanvas(keys='interactive', show=True)
    view   = canvas.central_widget.add_view()
    view.camera = 'turntable'
    
    if centering:
        center = skel.hull_coords.mean(0)
    else:
        center=np.asarray([0,0,0])

    plots = plot_hull(skel, subparts, center)
    
    if line:
        plots.extend(plot_line(skel, center))
    if mi or vc or sj:
        plots.extend(plot_objects(skel, mi, vc, sj, center))
    for obj in plots:
        view.add(obj)
    return view

def plots(ids, line=True, subparts=None, mi=True, vc=True, sj=True, centering=True):
    skel   = get_skels(ids)
    canvas = SceneCanvas(keys='interactive', show=True)
    view   = canvas.central_widget.add_view()
    view.camera = 'turntable'
    
    if centering:
        center = skel.hull_coords.mean(0)
    else:
        center=np.asarray([0,0,0])

    plots = plot_hull(skel, subparts, center)
    
    if line:
        plots.extend(plot_line(skel, center))
    if mi or vc or sj:
        plots.extend(plot_objects(skel, mi, vc, sj, center))
    for obj in plots:
        view.add(obj)
    return view