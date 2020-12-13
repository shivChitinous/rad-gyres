#!/usr/bin/python
#Modified from 
#Basic script to parse atomic coordinates of PDB files by Chaitanya Athale, IISER Pune 2016/09/26
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import seaborn as sns

def atomize(pdbfile,heteroatom=False):
    pdbfile.seek(0) #setting readlines to 0
    atoms = np.array([line for line in pdbfile.readlines()])
    atoms = pd.Series(atoms[[('ATOM' in line[:10]) or (('HETATM' in line[:10]) & heteroatom)
                             for line in atoms]]).str.split(expand=True)
    
    #.pdb files seem to have spacing errors and so we need to fix some things before we can use the numbers
    #fixing non 3 proteins with shifted tables
    mask = ((atoms.iloc[:,3].str.len()!=3) & (atoms.iloc[:,4].str.isnumeric()))
    atoms.loc[mask,5:] = np.array(atoms.loc[mask,4:10])

    #fix all the non 3 proteins
    mask = (atoms.iloc[:,3].str.len()!=3)
    if np.any(mask):
        joint = atoms.loc[mask,2:4].agg(''.join,axis=1).str.replace('\d+','').copy()
        atoms.loc[mask,4] = joint.str[-1]
        atoms.loc[mask,3] = joint.str[-4:-1]
        atoms.loc[mask,2] = joint.str[:-4]

    #fix the none-types due to sticking of B-factor and occupancy
    mask = (np.array(atoms[11])==None)
    if np.any(mask):
        joint = atoms.loc[mask,9:].replace({None:''}).agg(''.join,axis=1)
        atoms.loc[mask,11] = joint.str[-1]
        atoms.loc[mask,10] = joint.str[4:-1]
        atoms.loc[mask,9] = joint.str[:4]
        atoms.loc[mask]

    #giving column names
    cols = ['record','serial','name','residue','chain',
            'seq-pos','x','y','z','occupancy','B-factor','element']
    atoms.columns = cols
    
    #specifying data types
    atoms['serial'] = atoms['serial'].astype('int')
    atoms[['x','y','z','occupancy','B-factor']] = atoms[['x','y','z','occupancy',
                                                         'B-factor']].astype('float')
    
    atoms = atoms.sort_values(by=['serial'])
    return atoms

def plot_protein(structure,skeleton=True,all_atoms=False,title="",chain_no=slice(None),point=None,sphere=None):
    #get colours
    structure['element-id'],indx = structure['element'].factorize()
    clrs = sns.color_palette("Set1",np.size(indx))
    cm = cl.ListedColormap(clrs.as_hex())

    #define figure
    fig = plt.figure(figsize=(4.5,4.5))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    
    #plotting all atoms
    if all_atoms:
        ax.scatter3D(structure['x'], structure['y'], structure['z'],c=structure['element-id'], 
                     marker='.', cmap=cm,alpha=0.4)
    
    #plotting skeleton
    if skeleton:
        slic = ((structure['name']=='CA')|(structure['name']=='N')|(structure['name']=='C'))
        _,chains = structure['chain'].factorize()
        chain_clr = sns.color_palette("Set2",np.size(chains))
        
        for i,p in enumerate(chains[chain_no]):
            slicer = (slic&(structure['chain']==p))

            #termini
            ax.scatter3D(structure['x'][slicer].iloc[0], structure['y'][slicer].iloc[0], structure[slicer]['z'].iloc[0],
                         color=clrs[0],s=50,marker='X')
            ax.scatter3D(structure['x'][slicer].iloc[-1], structure['y'][slicer].iloc[-1], 
                         structure[slicer]['z'].iloc[-1],color=clrs[1],s=50,marker='X')

            #skeleton
            ax.scatter3D(structure['x'].loc[slicer], structure['y'].loc[slicer], structure['z'].loc[slicer]
                      ,s=5,color=chain_clr[i])
            ax.plot3D(structure['x'].loc[slicer], structure['y'].loc[slicer], structure['z'].loc[slicer],
                      alpha=0.5,linewidth=2,color=chain_clr[i])
    
    if point is not None:
        ax.scatter3D(point[0], point[1], point[2],
                      s=30,marker='o',color='k')
    
    if sphere is not None:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = sphere*np.cos(u)*np.sin(v)+point[0]
        y = sphere*np.sin(u)*np.sin(v)+point[1]
        z = sphere*np.cos(v)+point[2]
        ax.plot_surface(x, y, z, alpha=0.2, color='lightsteelblue')
        
    ax.set_box_aspect((1,1,1))
    set_axes_equal(ax)

    plt.show()


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
