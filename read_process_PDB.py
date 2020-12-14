#!/usr/bin/python
#Modified from 
#Basic script to parse atomic coordinates of PDB files by Chaitanya Athale, IISER Pune 2016/09/26
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot
import plotly.graph_objects as go

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

def plot_protein(structure,title="",point=None,sphere=None):
    
    #skeleton
    mask = (structure['name']=='CA')|(structure['name']=='N')|(structure['name']=='C')
    fig1 = px.line_3d(structure.loc[mask],x='x',y='y',z='z',color='chain')
    fig1.update_traces(line=dict(width=5),legendgroup="chain")
    
    #termini
    _,chains = structure['chain'].factorize(); val = True
    for i,c in enumerate(chains):
        slic = (mask) & (structure['chain']==c)
        fig_term = px.scatter_3d(structure.loc[slic].iloc[[0,-1]],x='x',y='y',z='z',color='element',
                             color_discrete_sequence=px.colors.qualitative.Dark24_r)
        if i>0: val = False
        fig_term.update_traces(marker=dict(size=3,symbol='x'),legendgroup="termini",showlegend=val)
        fig1.add_traces(fig_term.data)
    
    #all atoms
    fig2 = px.scatter_3d(structure,x='x',y='y',z='z',color='element',
                        color_discrete_sequence=px.colors.qualitative.D3)
    if sphere is None: fig2.update_traces(marker=dict(size=2),legendgroup="element",visible="legendonly")
    else: fig2.update_traces(marker=dict(size=2),legendgroup="element")

    fig2.add_traces(fig1.data)
    fig2.update_layout(title=title,legend=dict(orientation="h",x=0,y=1,title=dict(text="   Atom    Chain   Terminus",
                                                                                  side="top")))
    
    #COM
    if point is not None:
        fig3 = px.scatter_3d(pd.DataFrame(np.array([point]),columns=['x','y','z']),x='x',y='y',z='z',
                    color_discrete_sequence=px.colors.qualitative.Dark2_r)
        
        fig3.update_traces(marker=dict(size=5))
        fig2.add_traces(fig3.data)
        
    #Rg
    if sphere is not None:
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = sphere*np.cos(u)*np.sin(v)+point[0]
        y = sphere*np.sin(u)*np.sin(v)+point[1]
        z = sphere*np.cos(v)+point[2]

        fig4 = go.Figure(go.Surface(
            x = x,
            y = y,
            z = z,
            opacity=0.1
            ))
        
        fig4.update_traces(showscale=False)
        fig2.add_traces(fig4.data)

    fig2.show()
