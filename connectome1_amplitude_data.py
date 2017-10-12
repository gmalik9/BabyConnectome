import numpy as np
import scipy.stats as st
import pylab
import matplotlib.pyplot as plt
from matplotlib import patches
import networkx as nx
from networkx.readwrite import json_graph
import os
import json
import glob
import community
import itertools
from collections import defaultdict




def graph_render_all(H,p1,f,r0,r1,p2='graphs'):
    '''
    Graph Rendering Function
    Renders multiple graphs given in the list at once

    Input:
    ------
    H: (list (of NetworkX Graph Object)): The graph for which the community needs to be detected
    p1: (str): The master path for the directories to be created in
    f: (str): The name of the file, used in the name of the graph's rendered image as well
    r0: (int): The minima of the range in consideration for which the network is being constructed/analyzed
    r1: (int): The maxima of the range in consideration for which the network is being constructed/analyzed
    p2: (str): Optional. Default='graphs': The name of the directory in which the result from this subroutine is to be stored

    Output:
    -------
    returns: null
    Stores the graphs in the path given
    
    '''
    path=p1+p2+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i in range(0,len(H)):
        labels={}    
        for j in range(0,len(H[i])):
            #labels[H[i].nodes()[j]]=r'Res#'+str(H[i].nodes()[j])+':'+lab2[dg1[H[i].nodes()[j]]]
            #labels[H[i].nodes()[j]]=str(H[i].nodes()[j])+':'+lab3[lab2[dg1[H[i].nodes()[j]]]]
            #labels[H[i].nodes()[j]]=lab3[lab2[dg1[H[i].nodes()[j]]]]
            labels[H[i].nodes()[j]]=j
        pos=nx.spring_layout(H[i],k=0.15,iterations=100)
        val_map={72:0.3,
                 74:0.3,
                 372:0.3,
                 373:0.3,
                 374:0.3}
        values = [val_map.get(node, 0.25) for node in H[i].nodes()]
        #pos=nx.shell_layout(H[i])
        #pos=nx.random_layout(H[i])
        #nx.draw(H[i],with_labels=True)
        pylab.figure(1,figsize=(23,23))
        #nx.draw_networkx_nodes(H[i],pos,node_size=1000,node_color=values)
        #nx.draw_networkx_nodes(H[i],pos,node_size=values*4000,node_color=values)
        nx.draw_networkx_nodes(H[i],pos,nodelist=[x for x in val_map.keys() if x in H[i].nodes()],node_size=2000,node_color="#FF0000")
        nx.draw_networkx_nodes(H[i],pos,nodelist=list(set(H[i].nodes())-set(val_map.keys())),node_size=800,node_color="#A0CBE2")
        nx.draw_networkx_edges(H[i],pos)
        nx.draw_networkx_labels(H[i],pos,labels,font_color='orange',font_weight='bold',font_size=20)
        pylab.axis('off')
        pylab.savefig(path+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.png')
        #pylab.savefig(path+f[i].split('/')[-1]+'_'+str(p5)+'_type4.png')
        pylab.close()
        ##plt.axis('off')
        #plt.show()
        ##plt.savefig(path+f[i].split('\\')[1]+'_type4.png',bbox_inches='tight',figsize=[1000,1000])
        #plt.show()
        ##plt.close()


def graph_render(H,p1,f,r0,r1,p2='graphs'):
    '''
    Graph Rendering Function
    Renders single graph given
    
    Input:
    ------
    H: (NetworkX Graph Object): The graph for which the community needs to be detected
    p1: (str): The master path for the directories to be created in
    f: (str): The name of the file, used in the name of the graph's rendered image as well
    r0: (int): The minima of the range in consideration for which the network is being constructed/analyzed
    r1: (int): The maxima of the range in consideration for which the network is being constructed/analyzed
    p2: (str): Optional. Default='community': The name of the directory in which the result from this subroutine is to be stored

    Output:
    -------
    returns: null
    Stores the graph in the path given
    
    '''
    path=p1+p2+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    labels={}    
    for j in range(0,len(H)):
        #labels[H[i].nodes()[j]]=r'Res#'+str(H[i].nodes()[j])+':'+lab2[dg1[H[i].nodes()[j]]]
        #labels[H[i].nodes()[j]]=str(H[i].nodes()[j])+':'+lab3[lab2[dg1[H[i].nodes()[j]]]]
        #labels[H[i].nodes()[j]]=lab3[lab2[dg1[H[i].nodes()[j]]]]
        labels[H.nodes()[j]]=j
    pos=nx.fruchterman_reingold_layout(H,k=0.15,iterations=10) 
    #pos=nx.spring_layout(H,k=0.15,iterations=100)
    val_map={72:0.3,
             74:0.3,
             372:0.3,
             373:0.3,
             374:0.3}
    values = [val_map.get(node, 0.25) for node in H.nodes()]
    #pos=nx.shell_layout(H[i])
    #pos=nx.random_layout(H[i])
    #nx.draw(H[i],with_labels=True)
    pylab.figure(1,figsize=(23,23))
    #nx.draw_networkx_nodes(H[i],pos,node_size=1000,node_color=values)
    #nx.draw_networkx_nodes(H[i],pos,node_size=values*4000,node_color=values)
    nx.draw_networkx_nodes(H,pos,nodelist=[x for x in val_map.keys() if x in H.nodes()],node_size=2000,node_color="#FF0000")
    nx.draw_networkx_nodes(H,pos,nodelist=list(set(H.nodes())-set(val_map.keys())),node_size=800,node_color="#A0CBE2")
    #nx.draw_networkx_nodes(H,pos,nodelist=list(set(H.nodes())-set(val_map.keys())),node_size=800,node_color="#FF0000")
    nx.draw_networkx_edges(H,pos)
    nx.draw_networkx_labels(H,pos,labels,font_color='orange',font_weight='bold',font_size=20)
    pylab.axis('off')
    pylab.savefig(path+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.png')
    #pylab.savefig(path+f.split('/')[-1]+'_'+str(p5)+'_type4.png')
    pylab.close()
    ##plt.axis('off')
    #plt.show()
    ##plt.savefig(path+f[i].split('\\')[1]+'_type4.png',bbox_inches='tight',figsize=[1000,1000])
    #plt.show()
    ##plt.close()




def distribution_plot(dt_cc,p1,f,r0,r1,p2='/'):
    '''
    Function to plot and save the distribution, given a list of values

    Input:
    ------
    dt_cc: (list): The list of values of a random variable, in this case, could be clustering coefficient
    p1: (str): The master path for the directories to be created in
    f: (str): The name of the file, used in the name of the graph's rendered image as well
    r0: (int): The minima of the range in consideration for which the network is being constructed/analyzed
    r1: (int): The maxima of the range in consideration for which the network is being constructed/analyzed
    p2: (str): Optional. Default='/': The name of the directory in which the result from this subroutine is to be stored

    Output:
    -------
    Draws two kinds of images of distribution, one with histogram and the other without and stores them in the given file at a given location in png format 
    '''
    path=p1+p2+str(r0)+'_'+str(r1)+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    #plotting the normal distribution 
    h=sorted(dt_cc)
    fit = st.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

    pylab.plot(h,fit,'-o')

    pylab.hist(h,normed=True)      #use this to draw histogram of your data

    pylab.savefig(path+'hist_dist_clustering_'+f+'_'+str(r0)+'_'+str(r1)+'.png')                   #use may also need add this
    pylab.close()


    hmean = np.mean(h)
    hstd = np.std(h)
    pdf = st.norm.pdf(h, hmean, hstd)
    plt.plot(h, pdf,'-x') # including h here is crucial
    plt.savefig(path+'dist_clustering_'+f+'_'+str(r0)+'_'+str(r1)+'.png')                   #use may also need add this
    plt.close()


def community_detection(g,p1,f,r0,r1,p2='community'):
    '''
    Community Detection Subroutine


    Detects community, given the graph and saves the image of the commiunity partition graph. Works on the principle of maximizing modularity.
    After detecting the best possible partition, it also renders an image of the graph and stores it in the specified directory.

    Input:
    ------
    g: (NetworkX Graph Object): The graph for which the community needs to be detected
    p1: (str): The master path for the directories to be created in
    f: (str): The name of the file, used in the name of the graph's rendered image as well
    r0: (int): The minima of the range in consideration for which the network is being constructed/analyzed
    r1: (int): The maxima of the range in consideration for which the network is being constructed/analyzed
    p2: (str): Optional. Default='community': The name of the directory in which the result from this subroutine is to be stored

    Output:
    -------
    returns: null
    Creates a partition on the given graph based on M.E.J. Newman's modularity and stores them in the specified directory in .png format
    '''

    path=p1+p2+'/'
    if not os.path.exists(path):
        os.makedirs(path)    
    #community finding begins
    G = g
    labels=dict()

    #first compute the best partition
    partition = community.best_partition(G)

    #drawing
    size = float(len(set(partition.values())))
    #pos = nx.spring_layout(G)
    pos=nx.fruchterman_reingold_layout(g,k=1.0,iterations=10) 
    count = 0.
    
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 350,
                                    node_color = str(count / size))
        

    nx.draw_networkx_edges(G,pos, alpha=0.5)

    
    for j in range(0,len(g)):
        #labels[H[i].nodes()[j]]=r'Res#'+str(H[i].nodes()[j])+':'+lab2[dg1[H[i].nodes()[j]]]
        labels[g.nodes()[j]]=str(g.nodes()[j])#+':'+lab3[lab2[dg1[H[i].nodes()[j]]]]
    nx.draw_networkx_labels(g,pos,labels,font_size=10,font_weight='bold',font_color="#00AB66")
    plt.axis('off')
    #plt.savefig(path+f[i].split('\\')[1]+'_community.png')
    plt.savefig(path+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.png',bbox_inches="tight")
    #plt.savefig(path+'Fig-1_'+str(i+1)+'_'+(f[i].split('\\')[1]).split('_')[0]+'.png',bbox_inches="tight")
    plt.close()
    #plt.show()

    #community finding ends

    
    #plotting the trace

    # Run louvain community finding algorithm
    louvain_community_dict = partition

    # Convert community assignmet dict into list of communities
    louvain_comms = defaultdict(list)
    for node_index, comm_id in louvain_community_dict.iteritems():
        louvain_comms[comm_id].append(node_index)
    louvain_comms = louvain_comms.values()

    nodes_louvain_ordered = [node for comm in louvain_comms for node in comm]
    draw_adjacency_matrix(G, nodes_louvain_ordered,path+'trace/',f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1), [louvain_comms], ["blue"])





def draw_adjacency_matrix(G,nodes,path, filename, partitions=[], colors=[],node_order=None):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    if not os.path.exists(path):
        os.makedirs(path)


    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)
    
    #Plot adjacency matrix in toned-down black and white
    fig = plt.figure(figsize=(5, 5)) # in inches
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = plt.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)

    plt.imshow(adjacency_matrix, cmap="Greys", interpolation="none")
    #plt.show()
    plt.savefig(path+'trace_'+filename+'.png')
    plt.close()




def json_writer(g,nodes,coordinates,max_x_coord,max_y_coord,min_x_coord,min_y_coord,p1,f,r0,r1,p2='interactive_graphs'):
    '''
    Writes the graphs to a json file and then calls the subroutine html_writer which creates an html file with the generated json as input
    The json file is not dumped as is, there is another key called "size" is added to increase the size of the node, based on its degree

    Input:
    ------
    g: (NetworkX Graph Object): The graph for which the community needs to be detected
    nodes: (list): The list of nodes accepted for the creation of the graph
    p1: (str): The master path for the directories to be created in
    f: (str): The name of the file, used in the name of the graph's rendered image as well
    r0: (int): The minima of the range in consideration for which the network is being constructed/analyzed
    r1: (int): The maxima of the range in consideration for which the network is being constructed/analyzed
    p2: (str): Optional. Default='interactive_graphs': The name of the directory in which the result from this subroutine is to be stored

    Output:
    -------
    Writes the json file for given graph and calls another subroutine called html_writer, for writing the html file corresponding to the json file created
    returns:
        d: (JSON Graph Object(dict)): A dictionary containing the data formatted in json format
    '''

    path=p1+p2+'/type1/'
    if not os.path.exists(path):
        os.makedirs(path)
    

    # write json formatted data
    d = json_graph.node_link_data(g) # node-link format to serialize
    num_edges=g.number_of_edges()
    #add an attribute called size based on the degree of nodes
    for i in nodes:
        #d['nodes'][nodes.index(i)]['size']=5+(100*(g.degree([i]).values()[0])/num_edges)
        d['nodes'][nodes.index(i)]['size']=5+7*(g.degree([i]).values()[0])
        d['nodes'][nodes.index(i)]['x']=(float(coordinates['E'+str(i)][0]))
        d['nodes'][nodes.index(i)]['y']=(float(coordinates['E'+str(i)][1]))
        d['nodes'][nodes.index(i)]['fixed']='true'

    # write json
    filename=f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)
    json.dump(d, open(path+filename+'.json','w'))
    html_writer(path,filename)

    path=p1+p2+'/type2/'
    if not os.path.exists(path):
        os.makedirs(path)

    # write json formatted data
    d = json_graph.node_link_data(g) # node-link format to serialize
    num_edges=g.number_of_edges()
    #add an attribute called size based on the degree of nodes

    for i in nodes:
        #d['nodes'][nodes.index(i)]['size']=5+(100*(g.degree([i]).values()[0])/num_edges)
        d['nodes'][nodes.index(i)]['size']=200
        d['nodes'][nodes.index(i)]['x']=(float(coordinates['E'+str(i)][0]))
        d['nodes'][nodes.index(i)]['y']=(float(coordinates['E'+str(i)][1]))
        d['nodes'][nodes.index(i)]['fixed']='true'

    # write json
    filename=f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)
    json.dump(d, open(path+filename+'.json','w'))
    html_writer(path,filename)



    path=p1+p2+'/type3/'
    if not os.path.exists(path):
        os.makedirs(path)

    #write the json for graphs without isolated nodes
    sw=[g.copy()]
    for k in range(0,len(sw)):
        #break using nx.connected_component_subgraphs(graphname) here
        iso=nx.isolates(sw[k])
        for j in iso:
            sw[k].remove_node(j)

        #write sw to json here
        # write json formatted data
        d = json_graph.node_link_data(sw[k]) # node-link format to serialize
        #num_edges=sw[i].number_of_edges()
        #add an attribute called size based on the degree of nodes
        for i in nodes:
            #d['nodes'][nodes.index(i)]['size']=5+(100*(g.degree([i]).values()[0])/num_edges)
            #d['nodes'][nodes.index(i)]['size']=5+7*(g.degree([i]).values()[0])
            if i not in iso:
                try:
                    d['nodes'][nodes.index(i)]['size']=200
                    d['nodes'][nodes.index(i)]['x']=(float(coordinates['E'+str(i)][0]))
                    d['nodes'][nodes.index(i)]['y']=(float(coordinates['E'+str(i)][1]))
                    d['nodes'][nodes.index(i)]['fixed']='true'
                except IndexError:
                    a=1
                    #d['nodes'][nodes.index(i)]['fixed']='true'


        # write json
        filename=f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)
        json.dump(d, open(path+filename+'.json','w'))
        html_writer(path,filename)
        
    
    return d,ps


def html_writer(path,filename):
    '''
    Creates HTML files for the visualtiozation of the graphs.

    Creates HTML files for the visualtiozation of the graphs, by inserting the json file in source.
    DEPENDANT on 2 input files: html1.txt and html2.txt, containing the source of the visualization script, utilizing d3js library.
    CALLED BY json_writer (subroutine)

    Input:
    ------
    path: (str): The path at which the .html file is to be created
    filename: (str): The name of the file to be created in .html format

    Output:
    -------
    returns: null
    Writes the HTML files to the given directory, with the corresponding .json file inserted in source.
    
    '''
    h1=open('html1.txt','r')
    h2=open('html2.txt','r')
    ht1=h1.read()
    ht2=h2.read()
    h1.close()
    h2.close()

    htm=open(path+filename+'.html','w')
    htm.write(ht1)
    htm.write(filename+'.json')
    htm.write(ht2)
    htm.close()

def degree_writer(g,p1,f,r0,r1,p2='degree'):
    '''
    Writes degrees of all the nodes for a given graph.

    Writes the degree of the corresponding node to a file of the given graph, in a tab delimited format.

    Input:
    ------
    g: (NetworkX Graph Object): The graph for which the community needs to be detected
    p1: (str): The master path for the directories to be created in
    f: (str): The name of the file, used in the name of the graph's rendered image as well
    r0: (int): The minima of the range in consideration for which the network is being constructed/analyzed
    r1: (int): The maxima of the range in consideration for which the network is being constructed/analyzed
    p2: (str): Optional. Default='degree': The name of the directory in which the result from this subroutine is to be stored

    Output:
    -------
    returns: null
    Writes the corresponding degree(the number of nodes directly connected to the given node) for all the nodes of the given graph, in the specified file at the specified path, in a tab delimited fashion.
    
    '''
    path=p1+p2+'/'
    if not os.path.exists(path):
        os.makedirs(path)

    path2=p1+p2+'/distribution/'
    if not os.path.exists(path2):
        os.makedirs(path2)

    path3=p1+p2+'/log_log_distribution/'
    if not os.path.exists(path3):
        os.makedirs(path3)


    deg=g.degree()
    fl=open(path+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','a+')
    fl.write('Range: '+str(r0)+' - '+str(r1)+'\n\n')
    fl.write('Node\tDegree\n')
    for w in sorted(deg, key=deg.get, reverse=True):
        fl.write(str(w)+'\t'+str(deg[w])+'\n')
    fl.write('\n\n\n')
    fl.close()



    #draw the node degree distribution
    in_degrees  = deg # dictionary node:degree
    in_values = sorted(set(in_degrees.values())) 
    in_hist = [in_degrees.values().count(x) for x in in_values]
    plt.figure()
    plt.plot(in_values,in_hist,'ro-') # in-degree
    #plt.plot(out_values,out_hist,'bv-') # out-degree
    #plt.legend(['In-degree','Out-degree'])
    #plt.xlabel('Degree')
    #plt.ylabel('Number of nodes')
    #plt.title('Hartford drug users network')
    plt.savefig(path2+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.png')
    plt.close()

    #draw the log-log node degree distribution
    items = sorted(deg.items ())
    fig = plt.figure ()
    ax = fig.add_subplot (111)
    ax.plot([k for (k,v) in items], [v for (k,v) in  items ])
    ax.set_xscale('log')
    ax.set_yscale('log')
    #fig.savefig("degree_distribution.png")
    fig.savefig(path3+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.png')
    plt.close()

    

def analysis_metrics(g,p1,f,r0,r1,p2='analysis'):

    '''
    Writes the various metrics useful for analysis of the given graph.

    Writes the various metrics for the given graph that could be useful for analysis to different files.
    Generates a directory called analysis, under which two more directories are created. The directory structure created is as follows

    analysis/
        |
        |-pearson_correlations/

        |
        |-network_summary/

    The directory 'pearson_correlations' holds files with the pearson correlations of the Adjacency Matrix of the given graph, for the specified ranges
    
    The directory 'network_summary' holds the files with the metrics useful for the network analysis, for the specified ranges. The various analysis metrics considered are:
        Average Clustering Coefficient
        Average of Betweenness Centrality
        Average of Closeness Centrality
        Average of Degree Centrality
        Average Degree Connectivity
        Average Neighbor Degree
        Average Node Connectivity
        Clustering Coefficients at nodes
        Betweenness Centrality
        Closeness Centrality
        Degree Centrality
        Shortest Path
        Shortest Path Length for nodes
        Average Shortest Path Length
        Community Partition

    Input:
    ------
    g: (NetworkX Graph Object): The graph for which the community needs to be detected
    p1: (str): The master path for the directories to be created in
    f: (str): The name of the file, used in the name of the graph's rendered image as well
    r0: (int): The minima of the range in consideration for which the network is being constructed/analyzed
    r1: (int): The maxima of the range in consideration for which the network is being constructed/analyzed
    p2: (str): Optional. Default='degree': The name of the directory in which the result from this subroutine is to be stored

    Output:
    -------
    returns: null
    Writes the various metrics useful for analysis of the grpahs, given the ranges, in two separate directories.
        
        


    '''

    path=p1+p2+'/pearson_correlations/'
    if not os.path.exists(path):
        os.makedirs(path)
    pearson_corr=nx.degree_pearson_correlation_coefficient(g)
    fl=open(path+'pearson_correlation_'+f.split('/')[-1]+'_'+str(p5)+'.txt','a+')
    fl.write('Range: '+str(r0)+' - '+str(r1)+': ')
    fl.write(str(pearson_corr))
    fl.write('\n')
    fl.close()


    path=p1+p2+'/network_summary/'
    if not os.path.exists(path):
        os.makedirs(path)

    path2=path+'clustering_coefficient/'+str(r0)+'_'+str(r1)+'/'
    if not os.path.exists(path2):
        os.makedirs(path2)

    ffcc=open(path2+'clustering_coefficients_all_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','a+')
    ffcc.write(f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'\t'+str(nx.average_clustering(g))+'\n')
    ffcc.close()


    fl=open(path+'network_summary_'+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','w')
    bet_cen=nx.betweenness_centrality(g)
    close_cen=nx.closeness_centrality(g)
    deg_cen=nx.degree_centrality(g)
    fl.write('\n\nAverage Clustering Coefficient: '+str(nx.average_clustering(g)))
    fl.write('\n\nAverage of Betweenness Centrality: '+str(np.mean(bet_cen.values())))
    fl.write('\n\nAverage of Closeness Centrality: '+str(np.mean(close_cen.values())))
    fl.write('\n\nAverage of Degree Centrality: '+str(np.mean(deg_cen.values())))
    fl.write('\n\nAverage Degree Connectivity: '+str(nx.average_degree_connectivity(g)))
    fl.write('\n\nAverage Neighbor Degree: '+str(nx.average_neighbor_degree(g)))
    fl.write('\n\nAverage Node Connectivity: '+str(nx.average_node_connectivity(g)))
    fl.write('\n\nClustering Coefficients at nodes: '+str(nx.clustering(g)))
    fl.write('\n\nBetweenness Centrality: '+str(bet_cen))
    fl.write('\n\nCloseness Centrality: '+str(close_cen))
    fl.write('\n\nDegree Centrality: '+str(deg_cen))
    fl.write('\n\nShortest Path: '+str(nx.shortest_path(g)))
    fl.write('\n\nShortest Path Length for nodes: '+str(nx.shortest_path_length(g)))
    
    # writing the hubs and authority for each node
    hubs,auth=nx.hits(g)
    fl.write('\n\nHubs score: '+str(hubs))
    fl.write('\n\nAuthority score: '+str(auth))

    # calculate the number of hubs
    # as implemented in Poli et al. 2015
    # hub is a node having a degree value of at least one standard deviation above the network's mean value
    # calculating the one std dev above mean measure using scipy's zscore function 
    zs=st.zscore(nx.degree(g).values())
    hubs=[nh for nh in range(0,len(zs)) if zs[nh]>1]
    fl.write('\n\nHubs: '+str(hubs))
    fl.write('\n\nNumber of Hubs: '+str(len(hubs)))
   	

    if nx.is_connected(g):
        fl.write('\n\nAverage Shortest Path Length: '+str(nx.average_shortest_path_length(g)))
    else:
        fl.write('\n\nNo Average Shortest Path Length as the Graph is not connected')

    part = community.best_partition(g)
    fl.write('\n\nCommunity Partition: '+str(part))
    mod = community.modularity(part,G)
    fl.write("\n\nModularity: "+str(mod))
    fl.close()

    # writing edgelist to a separate file
    # added number of edges
    # if importing edgelist for graph reconstruction, 
    # read the file using readlines and consider only the first line, 
    # discard the rest of the file
    fl=open(path+'edges_'+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','w')
    fl.write(str(g.edges()))
    fl.write('\n\nNumber of edges: '+str(g.number_of_edges()))
    fl.close()


    fl=open(path+'cliques_recursive_'+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','w')
    fl.write('Number of Cliques of each node: '+str(nx.clique.number_of_cliques(g))+'\n\n')
    fl.write(str(list(nx.clique.find_cliques_recursive(g))))
    fl.close()



    #writing the adjacency matrix
    path=p1+p2+'/adjacency_matrix/'
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(path+'adjacency_matrix_'+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.csv',nx.to_numpy_matrix(g),delimiter=',')

    draw_adjacency_matrix(g, nodes,path+'trace/', f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1))



    #plotting the crcular graphs for the small world
    path=p1+p2+'/small_world/'
    if not os.path.exists(path):
        os.makedirs(path)

    path2=p1+p2+'/small_world/graphs/'
    if not os.path.exists(path2):
        os.makedirs(path2)

    path3=p1+p2+'/small_world/topology/'
    if not os.path.exists(path3):
        os.makedirs(path3)
        
    #path4 used for nodes writing 

    path5=p1+p2+'/small_world/clustering_coefficient/'+str(r0)+'_'+str(r1)+'/'
    if not os.path.exists(path5):
        os.makedirs(path5)
    
    
    sw=[g.copy()]
    ff3=open(path+'small_world_summary_report_0.3_'+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','w')
    ff5=open(path+'small_world_summary_report_0.5_'+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','w')
    ff7=open(path+'small_world_summary_report_0.7_'+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','w')
    ff=open(path+'small_world_summary_report_combined_'+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','w')
    ffreport=open(path+'small_world_summary_report_details_'+f.split('/')[-1]+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','w')

    
    ff.write('If sws>1, the network is a small world\n\n')
    ff3.write('If sws>1, the network is a small world\n\n')
    ff5.write('If sws>1, the network is a small world\n\n')
    ff7.write('If sws>1, the network is a small world\n\n')
    ff.write('PDB Name\tfor p=0.3\tfor p=0.5\tfor p=0.7')
    ffreport.write('PDB Name\tNodes\tNumber of Nodes\tNumber of Edges\tClustering Coefficient\tAvg. Shortest Path Length')
    for i in range(0,len(sw)):
        #break using nx.connected_component_subgraphs(graphname) here
        iso=nx.isolates(sw[i])
        if len(iso)>0:
            ff3.write('\n'+f.split('/')[-1]+' is not connected, removing the isolated nodes first\n')
            ff5.write('\n'+f.split('/')[-1]+' is not connected, removing the isolated nodes first\n')
            ff7.write('\n'+f.split('/')[-1]+' is not connected, removing the isolated nodes first\n')
            ff.write('\n\n'+f.split('/')[-1]+' is not connected, removing the isolated nodes first')
            ffreport.write('\n\n'+f.split('/')[-1]+' is not connected, removing the isolated nodes first')
        for j in iso:
            sw[i].remove_node(j)
        #zb=list(nx.connected_component_subgraphs(H[i]))
        zb=list(nx.connected_component_subgraphs(sw[i]))
        if len(zb)>1:
            ff3.write('\n'+f.split('/')[-1]+' can be broken into '+str(len(zb))+' subgraphs, breaking and proceeding\n')
            ff5.write('\n'+f.split('/')[-1]+' can be broken into '+str(len(zb))+' subgraphs, breaking and proceeding\n')
            ff7.write('\n'+f.split('/')[-1]+' can be broken into '+str(len(zb))+' subgraphs, breaking and proceeding\n')
            ff.write('\n\n'+f.split('/')[-1]+' can be broken into '+str(len(zb))+' subgraphs, breaking and proceeding')
            ffreport.write('\n\n'+f.split('/')[-1]+' can be broken into '+str(len(zb))+' subgraphs, breaking and proceeding')
        #plotting the small world graphs in circular after removing isolated nodes
        for k in range(0,len(zb)):
            labels={}    
            for j in range(0,len(zb[k])):
                #labels[H[i].nodes()[j]]=r'Res#'+str(H[i].nodes()[j])+':'+lab2[dg1[H[i].nodes()[j]]]
                #labels[H[i].nodes()[j]]=str(H[i].nodes()[j])+':'+lab3[lab2[dg1[H[i].nodes()[j]]]]
                labels[zb[k].nodes()[j]]=str(zb[k].nodes()[j])#lab3[lab2[dg1[zb[k].nodes()[j]]]]
            #pos=nx.spring_layout(H[i])
                
            #pos=nx.circular_layout(zb[k])
            pos=nx.fruchterman_reingold_layout(zb[k],k=1.0,iterations=10)
            #pos=nx.shell_layout(H[i])
            #pos=nx.random_layout(H[i])
            #nx.draw(H[i],with_labels=True)
            nx.draw_networkx_nodes(zb[k],pos,node_size=350)
            nx.draw_networkx_edges(zb[k],pos,alpha=0.5)
            nx.draw_networkx_labels(zb[k],pos,labels,font_size=10,font_weight='bold')
            plt.axis('off')
            #plt.show()
            plt.savefig(path2+f.split('/')[-1]+'_subset_'+str(k+1)+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.png',bbox_inches='tight')
            #plt.show()
            plt.close()

            #writing the nodes and degrees
            #fd=open(path3+'degree_'+f.split('/')[-1]+'_subset_'+str(k+1)+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','w')
            degree_writer(zb[k],path3,f+'_subset_'+str(k+1),r0,r1,p2='degree')
            #fd.write
            #fd.close()

            path4=path3+'nodes/'
            if not os.path.exists(path4):
                os.makedirs(path4)
                
            fn=open(path4+'nodes_'+f.split('/')[-1]+'_subset_'+str(k+1)+'_'+str(p5)+'_'+str(r0)+'_'+str(r1)+'.txt','w')
            fn.write(str(zb[k].nodes()))
            fn.close()



    
            
            GGG3=nx.erdos_renyi_graph(zb[k].number_of_nodes(),0.3)
            GGG5=nx.erdos_renyi_graph(zb[k].number_of_nodes(),0.5)
            GGG7=nx.erdos_renyi_graph(zb[k].number_of_nodes(),0.7)
    
            try:
                lamb3=nx.average_shortest_path_length(zb[k])/nx.average_shortest_path_length(GGG3)
                gama3=nx.average_clustering(zb[k])/nx.average_clustering(GGG3)
                sws3=gama3/lamb3
            except (nx.NetworkXError, ZeroDivisionError):
                ff3.write(f.split('/')[-1]+'\nGraph #'+str(k+1)+' cannot be compared with p=0.3')
    
            try:
                lamb5=nx.average_shortest_path_length(zb[k])/nx.average_shortest_path_length(GGG5)
                gama5=nx.average_clustering(zb[k])/nx.average_clustering(GGG5)
                sws5=gama5/lamb5
            except (nx.NetworkXError, ZeroDivisionError):
                ff5.write(f.split('/')[-1]+'\nGraph #'+str(k+1)+' cannot be compared with p=0.5')
    
            try:
                lamb7=nx.average_shortest_path_length(zb[k])/nx.average_shortest_path_length(GGG7)
                gama7=nx.average_clustering(zb[k])/nx.average_clustering(GGG7)
                sws7=gama7/lamb7
            except (nx.NetworkXError, ZeroDivisionError):
                ff7.write(f.split('/')[-1]+'\nGraph #'+str(k+1)+' cannot be compared with p=0.5')
    
    
            ff.write('\n\n'+f.split('/')[-1]+'_subset_'+str(k+1)+'\t')
            ffreport.write('\n\n'+f.split('/')[-1]+'_subset_'+str(k+1)+'\t')
            ffcc=open(path5+'subset_'+str(k+1)+'.txt','a+')
            ffcc.write(f.split('/')[-1]+'_subset_'+str(k+1)+'\t')
            ffcc.write(str(nx.average_clustering(zb[k]))+'\n')
            ffcc.close()

            try:
                ffreport.write(str([a for a in zb[k].nodes()])+'\t'+str(zb[k].number_of_nodes())+'\t'+str(zb[k].number_of_edges())+'\t'+str(nx.average_clustering(zb[k]))+'\t'+str(nx.average_shortest_path_length(zb[k]))+'\n')
            except ZeroDivisionError:
                ffreport.write('Only a single node\n')
                
            ff3.write(f.split('/')[-1]+'\nGraph #'+str(k+1))
            try:
                ff3.write('\nsws='+str(sws3))
                if sws3>1:
                    ff3.write('\nIt is a small world network\n')
                    ff.write('Small World\t')
                    
                else:
                    ff3.write('\nIt is not a small world network\n')
                    ff.write('Not Small World\t')
            except NameError:
                ff3.write('\nCombination not possible\n')
                ff.write('Not Possible\t')
    
            ff5.write(f.split('/')[-1]+'\nGraph #'+str(k+1))
            try:
                ff5.write('\nsws='+str(sws5))
                if sws5>1:
                    ff5.write('\nIt is a small world network\n')
                    ff.write('Small World\t')
                    
                else:
                    ff5.write('\nIt is not a small world network\n')
                    ff.write('Not Small World\t')
            except NameError:
                ff5.write('\nCombination not possible\n')
                ff.write('Not Possible\t')
                
            ff7.write(f.split('/')[-1]+'\nGraph #'+str(k+1))
            try:
                ff7.write('\nsws='+str(sws7))
                if sws7>1:
                    ff7.write('\nIt is a small world network\n')
                    ff.write('Small World\t')
                    
                else:
                    ff7.write('\nIt is not a small world network\n')
                    ff.write('Not Small World\t')
            except NameError:
                ff7.write('\nCombination not possible\n')
                ff.write('Not Possible\t')
    
            try:
                ff.write('\t\t'+str(sws3))
            except NameError:
                ff.write('\t\tNot Possible')
            try:
                ff.write('\t'+str(sws5))
            except NameError:
                ff.write('\tNot Possible')
            try:
                ff.write('\t'+str(sws7))
            except NameError:
                ff.write('\tNot Possible')
    
            #ff.write('\t\t'+str(sws3)+'\t'+str(sws5)+'\t'+str(sws7))
    
    ff3.close()
    ff5.close()
    ff7.close()
    ff.close()
    ffreport.close()




    
    return pearson_corr

def auto_averaged(corr1,corr2,name,p1,f,ranges,p2='auto_averaged'):
    '''
    Function to Calculate the average for the two ranges. Takes the two quantities and writes to a file their individual and cumulative average


    Input:
    ------
    corr1: (list): The list containing the metric for the first range
    corr2: (list): The list containing the metric for the second range
    name: (str): The name of the measure, used to create the file with that name
    p1: (str): The master path for the directories to be created in
    f: (str): The name of the first file. The second last string is used, after spliting on '/'.
    ranges: (list): The list of ranges being considered in the current run

    Output:
    -------
    returns: null
    Writes the average of the metrics for each range and also the cumulative average, for a directory, to a file specified by f

    
    '''
    path=p1+p2+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    fl=open(path+'averaged_'+name+'_'+f.split('/')[-2]+'_'+str(p5)+'.txt','w')
    fl.write('Range: '+str(ranges[0])+': '+str(np.average(corr1))+'\n')
    fl.write('Range: '+str(ranges[1])+': '+str(np.average(corr2))+'\n')
    fl.write('Range: '+'cumulative'+': '+str(np.average([corr1,corr2]))+'\n')
    fl.close()


def flatten(S):
    '''
    Flatten the given list recursively
    '''
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])



def range_reducer(dt,r0,r1):
    '''
    Function to reduce the data to a certain range 
    '''
    return dt[range(r0-1,r1),:]



def graph_maker(nodes,ranges,f,a,r,r0,r1):
    ##    for r in ranges:
    ##        r0=int(r.split('-')[0]) #the range begins here
    ##        r1=int(r.split('-')[1]) #the range ends here
    ##        a=dt[range(r0-1,r1),:]

    '''
    b=st.spearmanr(a[:,0],a[:,1])
    print b
    print st.spearmanr(a[:,1],a[:,0])

    print st.pearsonr(a[:,0],a[:,1])
    print st.pearsonr(a[:,1],a[:,0])
    '''
        

    avg=[] #avg is the variable holding the values for correlation coefficients with 95% significance. The name avg should not be confused that it is only being used for average here.
    for i in range(0,len(nodes)):
        for j in range(i+1,len(nodes)):
            b=st.spearmanr(a[:,nodes[i]],a[:,nodes[j]])
            if b[1]<sig:
                avg.append(b[0])
    average=np.average(avg)
    #print avg
    #print '\n\n'
    mx=max(avg)
    mn=min(avg)
    p5_mx=p5*mx
    p5_mn=p5*mn
    limit_mx=mx-p5_mx
    limit_mn=mn-p5_mn


    g=nx.Graph()
    g.add_nodes_from(nodes)



    path=p1+'correlations/'
    if not os.path.exists(path):
        os.makedirs(path)

    ff=open(path+"corr_test_sig_"+f.split('/')[-1]+'_'+str(sig)+"_p5_"+str(p5)+'_'+str(r0)+'_'+str(r1)+".txt",'w')
    for i in range(0,len(nodes)):
        for j in range(i+1,len(nodes)):
            ff.write("\nCorrelations between: "+str(nodes[i])+" , "+str(nodes[j])+": ")
            b=st.spearmanr(a[:,nodes[i]],a[:,nodes[j]])
            ff.write(str(b))
            if b[1]<sig:
                ff.write(" Yes ")
                #avg.append(b[0])
            else:
                ff.write(" No ")
            if b[0]>average:
                ff.write(" Greater ")
            else:
                ff.write(" Lesser ")

            #this restriction of taking the 5% or 10% correlations can be made more
            #robust by selecting only the <0.05 p-value correlations
            if limit_mx <= b[0] <= mx:
                ff.write(" More ")
                g.add_edge(nodes[i],nodes[j])
            elif mn <= b[0] <= limit_mn:
                ff.write(" Less ")
                g.add_edge(nodes[i],nodes[j])

            
    ff.close()

    
    return g



       

    

#p=glob.glob("/Users/gmalik9/Desktop/Genome Designer/Baby Connectome/Text data/Time series data/Infant 14 Puff/*.txt")
#folder_path="/Users/gmalik9/Desktop/Genome Designer/Baby Connectome/Text data/Text data/"
folder_path="/home/gxm020/Text data/"
folders=glob.glob(folder_path+'*') # see the * appended at the end
folder_name="results_unisensory_resting_fullterm_only_infant_range_1" #the name to be given to the folder containing results

#read the coordinate file and convert to dictionary
ps=np.loadtxt('geodesic_net_128_position.txt',dtype=str)
max_x_coord=max([float(ps[i,1]) for i in range(0,len(ps))])
max_y_coord=max([float(ps[i,2]) for i in range(0,len(ps))])
min_x_coord=min([float(ps[i,1]) for i in range(0,len(ps))])
min_y_coord=min([float(ps[i,2]) for i in range(0,len(ps))])
coordinates=dict()
for i in range(0,len(ps)):
    coordinates[ps[i][0]] = ps[i][1:]
#[{coordinates[ps[i][0]] = ps[i][1:]} for i in range(0,len(ps))]
#d2 = {value:key for key in d1 for value in d1[key]}
#coordinate={key:value for key in ps[0] for value in ps[1:]}


#ranges=['524-917','918-1965']
#ranges=['371-440','441-540']
ranges=['171-240','241-340']
#ranges=['1-180','541-699']

nodes=set(range(1,128))
#n_exl=set([1,8,14,17,44,49,56,57,63,64,69,74,82,89,95,100,114,120,121,125,126,127,128]) #nodes to be excluded
n_exl=set([17,44,49,56,63,69,74,82,89,95,100,108,114,120,125,126,127,128,129]) #nodes to be excluded
nodes=nodes-n_exl
nodes=list(nodes)

global_dt=0
global_counter=0
gHr1=[]
gHr2=[]
for folder in folders:
    
    #p=glob.glob(folder+"/*.txt") # see the / followed by *.txt at the end
    p=glob.glob(folder+"/*")
    p1=folder_name+'/all/'+folder.split('/')[-1]+'/'

    #make the code following this comment a function and then call the function individually with different files and store the result
    #also make a matrix, keep on adding the other matrices over it, average it in the end
    #keep storing the subsequent averages and then take a grand average at the end of the code, depending on the number of infants considered (aintain a variable or use loop counter for that) 



    Hr1=[] #the list holding all the graphs of range 1
    Hr2=[] #the list holding all the graphs of range 2
    pearson_corr1=[] #pearson correlations for the first range
    pearson_corr2=[] #pearson correlations for the second range

    adt=0 #the numpy ndarray, for holding the sum of all the data
    for fl in range(0,len(p)):
        f=p[fl]
        #f="/Users/gmalik9/Desktop/Genome Designer/Baby Connectome/Text data/Time series data/Infant 14/infant_14_seg_puff, 1.txt"
        dt = np.loadtxt(f)
	#dt = np.loadtxt(f,delimiter=',') #use for the frequency data
	dt=dt[range(0,700),:] #taking the minimum time range out of all the files
	print f
	print '\n\n'
        adt=adt+dt #the numpy ndarray, holding the sum of all the data

        #declaration of significance. Left here if a different value needs to be used for different files/ranges
        sig=0.01 #the significance value, for 95% significance, use 0.05 (5%)

        #declaration of p5. Left here if a different value needs to be used for different files/ranges
        p5=0.05 #the percentage from distribution to be considered 


        for r in ranges:
            r0=int(r.split('-')[0]) #the range begins here
            r1=int(r.split('-')[1]) #the range ends here

            a=range_reducer(dt,r0,r1) #a is the data (dt), reduced to the specified range
            G=graph_maker(nodes,ranges,f,a,r,r0,r1)

            pearson_corr=analysis_metrics(G,p1,f,r0,r1)

            if r==ranges[0]:
                Hr1.append(G)
                pearson_corr1.append(pearson_corr)
            elif r==ranges[1]:
                Hr2.append(G)
                pearson_corr2.append(pearson_corr)
            
            graph_render(G,p1,f,r0,r1)
            d,ps=json_writer(G,nodes,coordinates,max_x_coord,max_y_coord,min_x_coord,min_y_coord,p1,f,r0,r1)


            degree_writer(G,p1,f,r0,r1)

            community_detection(G,p1,f,r0,r1,p2='community')

            
    gHr1.append(Hr1)    
    gHr2.append(Hr2)


    global_dt=global_dt+adt
    global_counter=global_counter+fl+1
    p1=p1+'averaged/'
    #Global Averaged Analysis begins
    auto_averaged(pearson_corr1,pearson_corr2,'pearson_correlation',p1,p[0],ranges)


    f='/'.join(p[0].split('/')[:-1])

    
    for r in ranges:
        r0=int(r.split('-')[0]) #the range begins here
        r1=int(r.split('-')[1]) #the range ends here

        #*** You may call the graph_maker function here on the adt/fl here ***

        #*** May have to change the input of f to be given to the various functions ***
        
        a=range_reducer(adt/(fl+1),r0,r1) #a is the data (dt), reduced to the specified range
        G=graph_maker(nodes,ranges,f,a,r,r0,r1)

        pearson_corr=analysis_metrics(G,p1,f,r0,r1)

##        if r==ranges[0]:
##            Hr1.append(G)
##            pearson_corr1.append(pearson_corr)
##        elif r==ranges[1]:
##            Hr2.append(G)
##            pearson_corr2.append(pearson_corr)

        graph_render(G,p1,f,r0,r1)

        json_writer(G,nodes,coordinates,max_x_coord,max_y_coord,min_x_coord,min_y_coord,p1,f,r0,r1)

        degree_writer(G,p1,f,r0,r1)

        community_detection(G,p1,f,r0,r1,p2='community')
        


        if r==ranges[0]:
            sets = [set(x.edges()) for x in Hr1]
            setc=set.intersection(*sets)
            g=nx.Graph()
            g.add_edges_from(list(setc))
            f='/'.join(p[0].split('/')[:-1])
            graph_render(g,p1,f,r0,r1,'intersection_graphs')
            json_writer(g,g.nodes(),coordinates,max_x_coord,max_y_coord,min_x_coord,min_y_coord,p1,f,r0,r1,'interactive_intersection_graphs')
            degree_writer(g,p1,f,r0,r1)

        elif r==ranges[1]:
            sets = [set(x.edges()) for x in Hr2]
            setc=set.intersection(*sets)
            g=nx.Graph()
            g.add_edges_from(list(setc))
            f='/'.join(p[0].split('/')[:-1])
            graph_render(g,p1,f,r0,r1,'intersection_graphs')
            json_writer(g,g.nodes(),coordinates,max_x_coord,max_y_coord,min_x_coord,min_y_coord,p1,f,r0,r1,'interactive_intersection_graphs')
            degree_writer(g,p1,f,r0,r1)



# ***** use the functions with the global approach here *****
p1=folder_name+'/all/grand_average/averaged/'
f='grand_average'
for r in ranges:
    r0=int(r.split('-')[0]) #the range begins here
    r1=int(r.split('-')[1]) #the range ends here

    a=range_reducer(global_dt/global_counter,r0,r1) #a is the data (dt), reduced to the specified range
    G=graph_maker(nodes,ranges,f,a,r,r0,r1)

    pearson_corr=analysis_metrics(G,p1,f,r0,r1)

    if r==ranges[0]:
        Hr1.append(G)
        pearson_corr1.append(pearson_corr)
    elif r==ranges[1]:
        Hr2.append(G)
        pearson_corr2.append(pearson_corr)

    graph_render(G,p1,f,r0,r1)

    json_writer(G,nodes,coordinates,max_x_coord,max_y_coord,min_x_coord,min_y_coord,p1,f,r0,r1)

    degree_writer(G,p1,f,r0,r1)

    community_detection(G,p1,f,r0,r1,p2='community')



    '''
    chain = itertools.chain(*gHr1)
    gHr1=list(chain)

    chain2 = itertools.chain(*gHr2)
    gHr2=list(chain2)
    '''

    #gHr1=sum(gHr1,[])
    #gHr2=sum(gHr2,[])

    gHr1=flatten(gHr1)
    gHr2=flatten(gHr2)
    
    if r==ranges[0]:
        sets = [set(x.edges()) for x in gHr1]
        setc=set.intersection(*sets)
        g=nx.Graph()
        g.add_edges_from(list(setc))
        #f='/'.join(p[0].split('/')[:-1])
        f='grand_average'
        graph_render(g,p1,f,r0,r1,'intersection_graphs')
        json_writer(g,g.nodes(),coordinates,max_x_coord,max_y_coord,min_x_coord,min_y_coord,p1,f,r0,r1,'interactive_intersection_graphs')
        degree_writer(g,p1,f,r0,r1)

    elif r==ranges[1]:
        sets = [set(x.edges()) for x in gHr2]
        setc=set.intersection(*sets)
        g=nx.Graph()
        g.add_edges_from(list(setc))
        #f='/'.join(p[0].split('/')[:-1])
        f='grand_average'
        graph_render(g,p1,f,r0,r1,'intersection_graphs')
        json_writer(g,g.nodes(),coordinates,max_x_coord,max_y_coord,min_x_coord,min_y_coord,p1,f,r0,r1,'interactive_intersection_graphs')
        degree_writer(g,p1,f,r0,r1)




#node similarity percent matcher for small world 
#folder_name="results_unisensory_freq" #the name to be given to the folder containing results
avg_folder='node_percent_analysis_unisensory'


g_avg=glob.glob(folder_name+'/all/grand_average/averaged/analysis/small_world/topology/nodes/*.txt') #holds all the nodes from all the subsets of grand average
avg=glob.glob(folder_name+'/all/*/') #holds only the name of the infant folders


for i in range(0,len(avg)):
    p1=folder_name+'/'+avg_folder+'/'+avg[i].split('/')[-2]+'/' #look at the [-2], split will give last element as empty because of path ending in '/', therefore use -2, which gives infant number
    path=p1
    if not os.path.exists(path):
        os.makedirs(path)

    p=glob.glob(avg[i]+'averaged/analysis/small_world/topology/nodes/*.txt')

    for j in range(0,len(g_avg)):
        a=np.loadtxt(g_avg[j],dtype=str,delimiter=',')
        a[0]=a[0].replace('[','')
        a[-1]=a[-1].replace(']','')
        a=[int(a[ind]) for ind in range(0,len(a))]
        l=len(a)

        ff=open(path+'node_precent_'+g_avg[j].split('/')[-1],'w')
        ff.write('Compared to \t Nodes \t Common Nodes \t Percent\n')

        for k in range(0,len(p)):
            b=np.loadtxt(p[k],dtype=str,delimiter=',')
            b[0]=b[0].replace('[','')
            b[-1]=b[-1].replace(']','')
            b=[int(b[ind]) for ind in range(0,len(b))]

            ff.write(p[k].split('/')[-1]+' \t '+str(b)+' \t ')
            c=set(a).intersection(set(b))
            ff.write(str(c)+' \t ')
            ff.write(str(float(len(c))/float(l)*100))
            ff.write('\n')

        ff.close()





#clustering coefficient distribution plotter, dependancy: sub-routine distribution_plot
net_sum='analysis/network_summary/clustering_coefficient/'
sm_wrld='analysis/small_world/clustering_coefficient/'


#g_avg=glob.glob(folder_name+'/all/grand_average/averaged/analysis/small_world/topology/nodes/*.txt') #holds all the nodes from all the subsets of grand average
avg=glob.glob(folder_name+'/all/*/') #holds only the name of the infant folders
try:
    avg=[avg.remove(avg[i]) for i in range(0,len(avg)) if 'grand_average' in avg[i]]
except IndexError:
    dummy=1


for r in ranges:
    r0=int(r.split('-')[0]) #the range begins here
    r1=int(r.split('-')[1]) #the range ends here

    for i in range(0,len(avg)):
        #use the clustering coefficient for the normal one here, first do it for the entire network, then the small world
        p1=folder_name+'/'+avg_folder+'/'+avg[i].split('/')[-2]+'/' #look at the [-2], split will give last element as empty because of path ending in '/', therefore use -2, which gives infant number

        cc_path=glob.glob(avg[i]+net_sum+str(r0)+'_'+str(r1)+'/'+'*.txt')
        cc_ns=np.loadtxt(cc_path[0],dtype=str,delimiter='\t')
        cc_ns=cc_ns[:,1]
        cc_ns=[float(cc_ns[ind]) for ind in range(0,len(cc_ns))]
        #print cc_ns
        distribution_plot(cc_ns,p1+net_sum,avg[i].split('/')[-2],r0,r1)
        #print len(cc_ns[0])

        #drawing dist for small world, subset 1 clustering coefficient
        cc_path=glob.glob(avg[i]+sm_wrld+str(r0)+'_'+str(r1)+'/'+'*.txt')
        cc_int_path=[cc_path[ind] for ind in range(0,len(cc_path)) if 'subset_1' in cc_path[ind]]
        cc_path=cc_int_path
        cc_sw=np.loadtxt(cc_path[0],dtype=str,delimiter='\t')
        cc_sw=cc_sw[:,1]
        cc_sw=[float(cc_sw[ind]) for ind in range(0,len(cc_sw))]
        #print cc_ns
        distribution_plot(cc_sw,p1+sm_wrld,avg[i].split('/')[-2]+'_'+cc_path[0].split('/')[-1].replace('.txt',''),r0,r1)
        #print len(cc_ns[0])


    cc_ns=[]
    cc_sw=[]

    p1=folder_name+'/'+avg_folder+'/all_all/'

    for i in range(0,len(avg)):        
        cc_path=glob.glob(avg[i]+'averaged/'+net_sum+str(r0)+'_'+str(r1)+'/'+'*.txt')
        cc_int_ns=np.loadtxt(cc_path[0],dtype=str,delimiter='\t')
        cc_ns.append(float(cc_int_ns[1]))

        #drawing dist for small world, subset 1 clustering coefficient
        cc_path=glob.glob(avg[i]+'averaged/'+sm_wrld+str(r0)+'_'+str(r1)+'/'+'*.txt')
        cc_int_path=[cc_path[ind] for ind in range(0,len(cc_path)) if 'subset_1' in cc_path[ind]]
        cc_path=cc_int_path
        cc_int_sw=np.loadtxt(cc_path[0],dtype=str,delimiter='\t')
        cc_sw.append(float(cc_int_sw[1]))

    distribution_plot(cc_ns,p1+'averaged/'+net_sum,'all_infants',r0,r1)
    distribution_plot(cc_sw,p1+'averaged/'+sm_wrld,'all_infants_'+cc_path[0].split('/')[-1].replace('.txt',''),r0,r1)

      
            


'''
#plotting the normal distribution 
h=sorted(avg)
fit = st.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

pylab.plot(h,fit,'-o')

pylab.hist(h,normed=True)      #use this to draw histogram of your data

pylab.show()                   #use may also need add this


hmean = np.mean(h)
hstd = np.std(h)
pdf = st.norm.pdf(h, hmean, hstd)
plt.plot(h, pdf,'-x') # including h here is crucial
plt.show()

'''



'''
#Test for range inclusion
if -0.96 <= -0.95 <= -0.91:
    print 1
else:
    print "No"

	
if 0.91 <= 0.95 <= 0.96:
    print 1
else:
    print "No"
'''

	
