#!/usr/bin/python
"""
**Moduler Copyright (C) 2012  Jose Sergio Hleap, with contributions of Kyle Nguyen, Alex Safatli and Christian Blouin**

Graph based Modularity.
This script will evaluate the data for modules. Such modules are defined as correlating variables, so the clustering 
is performed in the correlation space. It has an optional statistical significance test for the clustering and power
analysis of the result, as well as a bootstrap analysis. See options for more details.


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

E-mail: jshleap@dal.ca


In this version the the R dependencies have been extracted, and with them the RV coefficient test.

Requirements:
-------------

1. numpy module
2. scipy module
3. statsmodels module
4. matplotlib
5. scikit-learn   
6. PDBnet module: This is an open source module and can be found in :download: `LabBlouinTools<https://github.com/LabBlouin/LabBlouinTools>`

 *To install python modules 1-5 in UBUNTU: sudo apt-get install python-<module>	OR  sudo easy_install -U <module>  
 OR sudo pip install -U <module>*

 *For PDBnet, the whole directory should be downloded into a labblouin folder which should be in your pythonpath*

Citation
========

If you use this software, please cite: 
Hleap, J.S., Susko, E., & Blouin, C. Defining structural and evolutionary modules in proteins: a community detection approach to explore sub-domain architecture. BMC Structural Biology, 13, 20.

If you use the LDA pre-merge approach, please cite:
Hleap, J.S. & Blouin, C. 2014. Inferring meaningful communities from topology constrained correlation networks. PLOS ONE 9(11):e113438. DOI:10.1371/journal.pone.00113438
"""
__author__        = ['Jose Sergio Hleap',]
__contributions__ = ['Kyle Nguyen', 'Alex Safatli', 'Christian Blouin']
__version__       = 1.1

# Standard Library Imports
from numpy import array, cov, corrcoef, zeros, mean, std, sqrt, savetxt, \
     column_stack, isnan, rad2deg, arccos, where, concatenate, triu_indices,\
     log, trace, arange, percentile, arctanh, triu, mat, sum, subtract,\
     vstack, equal, nonzero, savetxt, ndarray
from numpy.random import choice
from numpy.random import rand as randm
from numpy.linalg import eig, pinv
from scipy.stats import pearsonr, kendalltau, spearmanr, norm
from igraph import Graph, VertexClustering
from os.path import isfile,dirname,join
from matplotlib.patches import Ellipse
from joblib import Parallel, delayed
from collections import Counter
from copy import deepcopy
from random import sample
from os import rename
from glob import glob
import optparse
import datetime
import sys
try:import dill as pickle
except: import pickle

# Third Party Imports
import statsmodels.stats.power as smp
from sklearn.lda import LDA

# Library-Specific Imports
from labblouin.PDBnet import PDBstructure as P




# __licence__       = open(join(dirname(__file__),'gpl.txt')).read()
# main classes
class GMdata:
    ''' 
    GM object that populates GM data

    :param prefix: Prefix of your GM file and accompanying files
    :type prefix: string
    :param dimension: The number of cartesian dimensions to be analyzed. By default is set to 3.
    :type dimension: integer
    :param t: Type of input. Either gm or csv, both semicolon delimited files, but the former includes names of the observations in the first field. By default is gm
    :type t: string
    :param asdistance: whether or not to tranform the data into a distance matrix
    :type asdistance: boolean
    :param Matrix: user defined matrix. By default a file will be read, based on t.
    :type Matrix: :class `numpy.array`
    '''	
    def __init__(self, prefix, dimension = 3, t='gm',asdistance=False, contacts=False, 
                 Matrix=False):
        self.dim = dimension
        self.prefix = prefix
        self.sample_size = None
        self.type = t
        if isfile(self.prefix+'.pdb'): self.pdb = P(self.prefix+'.pdb')
        if not isinstance(Matrix,bool):
            self.data = []
            for d in xrange(self.dim):
                self.data.append(Matrix[:,d::self.dim])
            self.data = array(self.data)
        else:
            # read file
            self.Read_GM()
        # transform to distance?
        if asdistance:
            asdistance(self)
        if contacts:
            self.Load_contacts()
        else:
            self.contacts = False


    def Read_GM(self):
        '''
        Load data from a gm (coordinates file) file. The file is a semicolon 
        separated file with row names in the first field.
        '''
        # Load Data ##########################################################
        data = []
        sample_size = 0
        labels = []
        # line for each dimension and create slush temp lists
        temps = []
        for i in range(self.dim):
            temps.append([])
            data.append([])
        # Set the filename with the proper suffix
        if self.type == 'csv':
            fname = self.prefix+'.csv'
        else:
            fname = self.prefix+'.gm'
        # Open file and read each line
        with open(fname) as F:
            for line in F:
                #get the names
                labels.append(line.strip().split(';')[0])            				
                # Split into list
                if self.type == 'gm':
                    if line.strip()[-1] == ';':
                        line = line.strip().split(';')[1:-1]
                    else:
                        line = line.strip().split(';')[1:]
                elif self.type == 'csv':
                    if line.strip()[-1] == ';':
                        line = line.strip().split(';')[:-1]
                    else:
                        line = line.strip().split(';')

                # check if there is a titleline
                if all([not isFloat(x) for x in line]):
                    continue
                # Dispatch values in the right dimension list using 
                #i%dim to index
                for i in range(len(line)):
                    temps[i%self.dim].append(float(line[i]))
                # Grow each matrix by 1 row
                for i in range(len(temps)):
                    data[i].append(temps[i])
                sample_size += 1 
                # Flush temps 
                temps = []
                for i in range(self.dim):
                    temps.append([])

        self.data= array(data)
        self.sample_size = sample_size    
        self.labels =labels

    def bootstrap_replicate(self):
        '''
        Create a bootstrap replicate of the data
        '''
        temp = []
        ch   = choice(self.data.shape[1],self.data.shape[1])
        for d in xrange(self.dim):
            temp.append(self.data[d][ch,:])
        self.boot = array(temp)
        return self.boot

    def Load_contacts(self):
        '''
        Load the contacts from file, from PDBstructure or None
        '''
        self.contacts=[]
        fn = '%s.contacts'%(self.prefix)
        if isfile(fn):
            with open(fn) as F:
                for line in F:
                    l = line.strip().strip('(').strip(')').split(',')
                    self.contacts.append(tuple(l))
        elif isfile('%s.pdb'%(self.prefix)):
            if not self.pdb: self.pdb = P('%s.pdb'%(self.prefix))
            contacts = self.pdb.Contacts(fasta='%s.fasta'%(self.prefix))
            if isinstance(contacts,list): self.contacts = contacts
            elif isinstance(contacts,dict):
                c=[]
                for v in contacts.values():
                    if not c: c=v
                    else:
                        sc = set(c)
                        c  = list(sc.union(v))
                self.contacts = c
            self.pdb = pdb
            with open(self.prefix+'.contacts','w') as F: 
                F.write('\n'.join([str(x) for x in self.contacts]))
        else:
            raise Exception('No Contact file nor PDB file provided')

    def Correlated(self,GMstatsInstance):
        '''
        Include a GMstats instance

        :param GMstatsInstance: An instance of the class GMstats
        :type GMstatsInstance: :class GMstats
        '''
        self.GMstats = GMstatsInstance


###############################################################################
class GMstats:
    ''' 
    Include all stats related things with a GM file

    :param prefix: Prefix of your GM file and accompanying files
    :type prefix: string
    :param matrix: A numpy nd array with the coordinates or info to be analysed. It contains the dimensions as first element, rows and columns follow.
    :type matrix: :class `numpy.array`
    :param dimensions: The number of cartesian dimensions to be analyzed. By default is set to 3.
    :type dimensions: integer
    :param sample_size: Number of observations
    :type sample_size: integer
    '''	
    def __init__(self,prefix,matrix,dimensions,lms=None,sample_size=None,rand=True):
        self.data   = matrix
        self.prefix = prefix
        self.dim    = dimensions
        self.lms    = lms
        self.rand   = rand
        if sample_size == None: self.n = self.data.shape[0]
        else: self.n= sample_size
        if self.lms == None: 
            self.Compute_correlations()


    def Compute_correlations(self, method='fisher', absolutecov=False, 
                             confval=0.95, threshold=0.0, usecov=False,
                             writemat=False,additive=True,power=0.8):
        '''
        :param method: Which method to use for correlation. By default it uses the fisher transformation. Available options are pearson, spearman and fisher.        
        :type method: string
        :param confval: Define the confidence level (1-alpha) for the correlation test. By default is 0.95. Can take any float value between 0 and 1.
        :type confval: float
        :param absolutecov: Whether or not to use absolute values of the correlation matrix
        :type absolutecov: boolean
        :param power: Perform a statistical power analysis with this value as the desired Power (1-type II error). By default 0.8 is used. Can be any float from 0 to 1
        :type power: float
        :param usecov: Use covariance instead of correlation.
        :type usecov: boolean
        :param writemat: Write correlation/covariance matrices to file. By default is false. It can take a False, for the aggreatated matrix or cor for the correlation matrix.
        :param additive: Use the mean of the correlation in each dimension instead of the euclidean distance to aglomerate the dimensions. The default behaviour is additive.
        :type additive: boolean
        '''
        self.method      = method
        self.confval     = confval 
        self.absolutecov = absolutecov
        self.power       = power
        self.threshold   = threshold
        self.usecov      = usecov
        self.writemat    = writemat
        self.additive    = additive
        thr = 0
        # Create a correlation matrix testing for significance ################
        if self.method:
            self.Sigcorr()
        # Else, use covariance matrix instead or correlation without 
        # statistical test ####################################################
        elif self.usecov:
            thr = self.UseCov()
        else:
            thr = self.UseCorr()
        #self.matrices = self.matrices
        if mat == 'cor':
            for m in range(len(self.matrices)):
                savetxt(self.prefix+str(m)+'.mat',self.matrices[m],
                        delimiter=';', fmt='%s')

        # Agglomerate landmark dimensions #####################################
        if self.additive:
            self.agglomerare_additive()
        else:
            self.agglomerare_mean()

        if mat == 'agg':
            savetxt(self.prefix+'.lms',self.lms,delimiter=';',fmt='%s')		


    def agglomerare_additive(self):
        '''
        Agglomerate landmark dimensions using euclidean distance
        '''
        lms = []
        for i in range(0,self.matrices[0].shape[0]):
            temp = []
            for j in range(0, self.matrices[0].shape[0]):
                sm = 0.0
                # Sum over all dimensions
                for m in self.matrices:
                    d = m[i][j]
                    if self.absolutecov:
                        d = abs(d)
                    sm += (d**2)
                sq = sqrt(sm)
                temp.append(sq)
            lms.append(temp)	

        self.lms = array(lms)

    def agglomerare_mean(self):
        '''Agglomerate landmark dimensions using average of correlation	'''
        lms = []
        for i in xrange(0,self.matrices[0].shape[0]):
            temp = []
            for j in xrange(0, self.matrices[0].shape[0]):
                sm = 0.0
                # Sum over all dimensions
                for m in self.matrices:
                    d = m[i][j]
                    if self.absolutecov:
                        d = abs(d)
                    sm += (d/self.dim)

                temp.append(sm)
            lms.append(temp)	

        self.lms = array(lms)

    def Sigcorr(self):
        '''
        Test if the correlation is significantly different than 0 with the method specified
        '''
        if isbetterthanrandom(self.dim,self.data):
            self.matrices = []
            if self.dim == 1:
                self.matrices.append(self.SigCorrOneMatrix(self.data))
            else:
                for i in xrange(len(self.data)):
                    # Perform the significant correlation test in each dimension
                    self.matrices.append(self.SigCorrOneMatrix(self.data[i]))
        else:
            sys.exit("Your data's correlation is not better than random. Exiting Moduler")

    def SigCorrOneMatrix(self, sliced):
        '''
        Performs the significance of correlation test according to the method passed

        :param sliced: an array with single dimensional data
        :type sliced: :class `numpy.array`
        '''
        data = sliced
        # Get a matrix of zeroes.
        zero=zeros(shape=(data.shape[1], data.shape[1]))

        for e in range(len(data.T)):
            for f in range(e, len(data.T)):
                if self.method == 'pearson' or self.method == 'fisher':
                    p=pearsonr(array(data.T[e]).ravel(),array(data.T[f]).ravel())
                if self.method == 'kendall':
                    p=kendalltau(array(data.T[e]).ravel(),array(data.T[f]).ravel())
                if self.method == 'spearman':
                    p=spearmanr(array(data.T[e]).ravel(),array(data.T[f]).ravel())					

                # Symmetrize	
                if self.method == 'fisher':
                    if p[0] == 1.0:
                        p = (0.999,p[1])
                    if self.absolutecov:
                        if abs(self.F_transf(p[0])) > self.Z_fisher()\
                           and self.Power_r(p[0]) <= self.n:
                            zero[e][f] = zero[f][e] = abs(p[0]) 
                        else:
                            zero[e][f] = zero[f][e] = 0.0

                    elif self.F_transf(p[0]) > self.Z_fisher()\
                         and self.Power_r(p[0]) <= self.n:
                        zero[e][f] = zero[f][e] = p[0]
                    else:
                        zero[e][f] = zero[f][e] = 0.0
                else:
                    if (p[1]<= (1-self.confval)) and (self.Power_r(p[0]) <= self.n):
                        zero[e][f] = zero[f][e] = p[0] 
                    else:
                        zero[e][f] = zero[f][e] = 0.0
        return zero	

    def F_transf(self,r):
        '''
        Compute the Fisher transformation of correlation
        '''	
        return 0.5 * (log((1+r)/(1-r)))

    def Z_fisher(self):
        '''
        Compute the sample - corrected Z_alpha for hypotesis testing of Fisher transformation of correlation
        '''		
        confval = 1-((1-self.confval)/2)
        return norm.isf(1-confval) / sqrt(self.n-3)

    def Power_r(self, corr):
        '''
        Compute the power of the correlation using the Z' trasnformation of correlation coefficient:
        Z'=arctang(r)+r/(2*(n-1)) (see Cohen (1988) p.546). 
        It will return the required n fo the power and significance chosen.

        :param corr: Correlation value
        :type corr: float
        '''

        if corr  == 0.0 or (corr < 0.01 and corr > -0.001) :
            corr = 0.0012
        elif corr < -0.001:
            corr =  corr
        elif corr >= 0.99:
            corr =  0.99			
        else:
            corr = corr	
        '''		
		r('p=pwr.r.test(power=%f,r=%f, sig.level=%f)'%(float(self.power), float(corr), float(1-self.confval)))
		n = r('p[1]')[0]
		if this part is reverted:
		Require the R package PWR written by Stephane Champely <champely@univ-lyon1.fr>. 
		'''
        #This is in beta to avoid dependencies outside python
        n = smp.NormalIndPower().solve_power(arctanh(corr), alpha=float(1-self.confval), ratio=0, 
                                             power= float(self.power), alternative='two-sided') + 3		

        return n	

    def UseCov(self):
        '''
        Create a variance-covariance matrix
        '''
        thr=[]
        # Perform np magic
        for i in range(len(data)):
            l=88
            # Convert to np matrix
            self.matrices.append(matrix(data[i]))
            l+=i
            self.matrices[i] = cov(self.matrices[i].T)
            if self.threshold == 'Auto':
                #get the threshold from correlation matrix#
                for e in range(len(self.matrices[i])):
                    thr.append(std(self.matrices[i][e]))
        return thr

    def UseCorr(self):
        '''
        Use pearson correlation without a significant test
        '''
        thr=[]
        # Perform np magic
        for i in range(len(data)):
            l=88
            # Convert to np matrix
            self.matrices.append(matrix(data[i]))
            l+=i
            self.matrices[i] = corrcoef(self.matrices[i].T)
            if self.threshold == 'Auto':
                #get the threshold from correlation matrix#
                for e in range(len(self.matrices[i])):
                    thr.append(std(self.matrices[i][e]))	
        return thr


    def GetAglomeratedThreshold(self):
        ''' Get threshold for the n dimensions'''
        # Call if you need to get the vector of threshold
        thr = []
        for i in range(len(self.lms)):
            for j in range(len(self.lms)):
                if i !=j and i < j and self.lms[i][j] != 0.00000:
                    thr.append(self.lms[i][j])
        threshold = std(thr)
        self.threshold = threshold 


    def LDA(self, membership, which='lms', ellipses=2.5):
        '''
        Perform a Linear discriminant analysis of the transposed data. Membership must be an array
        of integers of the same lenght of the number of observations in the data. 

        :param membership: a list corresponding to memberships of the entries in data
        :type membership: list
        :param which: Either 'gm' or 'lms'. To perform the LDA in the full matrix or only in the correlation matrix.
        :type which: string
        :param ellipses: A value representing the estandard deviations for confidence ellipses. By default is set to 2.5 (95% confidence ellipses)
        :type ellipses: float
        '''
        if which == 'gm': 
            data = self.data
            if len(data.shape) == 3:
                membership = membership*data.shape[0]
                data = column_stack(tuple([x for x in data]))
            elif any(((data.shape[0] == len(membership)), (data.shape[1] == len(membership)))):
                m = []
                membership = [m.extend([membership[i]]*self.dim) for i in xrange(len(membership))]
        else: data = self.lms

        membership = array(membership)
        #dont include singletons
        nonsingles = where(membership != '?')
        membership = membership[nonsingles]
        if len(set(membership)) >= 2:
            data       = data[nonsingles]
            lda = LDA(n_components=2)
            self.fit = lda.fit(data.T, membership).transform(data.T)
            if ellipses:
                ell = getEllipses(self.fit, membership,ellipses)
        else:
            print 'Not enough groups to perform LDA prefiltering. Exiting the program.'
            sys.exit()


    def Clus_Power(self, membership):
        '''
        Make a power analysis of the clusters by igraph and outputs a table with the proportion 
        of elements above the n required according to significance, power and correlation.
        it can test different clusters that are not in the class

        :param m: a membership vector in list form or the name of a file.
        :type m: list or string

        '''
        if isinstance(membership,str): mem = open(prefix+'.graphcluster').read().strip().split()
        else: mem = membership
        # alternative dictionary
        proportion={}
        # Split the data into the clusters ignoring singletons. This assumes that singletons have 
        # been identifed
        darr = clus2list(membership,self.matrices)
        if darr:
            # get all outputs
            resline = '#################################################################\n'
            resline+= '#Power analysis and description of the intracluster correlation:#\n'
            resline+= '#################################################################\n\n'
            resline+= '1 - Significance(alpha)= %f;\t1 - Power(beta):%f\n'%(1-self.confval,1-self.power)
            resline+= 'Sample Size : %d\n\n'%(self.n)
            #loop over clusters
            for clusters in darr:
                c=darr[clusters]
                # if is a list of n dimensions zip each correlation and return the max
                if isinstance(c,list): c=array([max(x) for x in zip(*c)])
                # estimate quantiles
                q = percentile(c, [0,25,50,75,100], axis=None, out=None, overwrite_input=False)
                # estimate the proportion of variables with enough power
                enough=0.0
                for x in c:
                    nsam = self.Power_r(x)
                    if int(nsam) <= self.n:
                        enough += 1.0
                prop = enough/len(c)
                tr = ' 0% \t\t 25% \t\t 50% \t\t 75% \t\t 100% \t\t PVP \t\t nvar \n'
                resline+= tr +' %f \t %f \t %f \t %f \t %f \t %f \t %d/%d \n\n'%(round(q[0],3),round(q[1],3),
                                                                                 round(q[2],3),round(q[3],3),round(q[4],3),
                                                                                 round(prop,3), enough, len(c))
                proportion[clusters]=prop
            resline+= '_'*len(tr)+'\n'
            resline+= 'Percentages are the quantiles of the lower triangle of the correlation matrix\n'
            resline+= 'PVP: Proportion of variables with enough statistical power \n'
            resline+= 'nvar: Number of variables with enough power within the cluster \n'
            # Open the output file and write resline
            # write to file
            with open('%s_Power_analysis.asc'%(self.prefix),'w') as ouf:
                ouf.write(resline)
            print resline
        else:
            print 'Only singletons in the dataset. Either correlation zero or not enough power to resolve'\
                  'a negligible correlation.'

        self.PVP = proportion
        return proportion	

###############################################################################################
class GMgraph:
    ''' 
    Create a graph based on an input in matrix form and compute modularity

    :param Matrix: an square matrix where the indices represent intended nodes and the values the relationship between them
    :type Matrix: :class `numpy.array` or :class `GMstats`
    :param unweighted: create an unweighted graph as opposed to a weighted (correlation; default) one.
    :type unweighted: boolean
    :param gfilter: List of tuples corresponding to desired conection of the graph. Each element in pair tuple must correspond to node indices in the graph. This is a topology constraint.
    :type gfilter: list of tuples
    :param threshold: A float corresponding to the threshold to create a conection between nodes. This is very user dependendent and is set to 0 as default.
    :type threshold: float
    :param rand: wether to test if your data is better than random. Default is true
    :type rand: boolean
    '''	
    def __init__(self, prefix, Matrix, unweighted=False, gfilter=[], threshold=0.0, rand=True):



        if isinstance(Matrix,GMstats): self.lms = Matrix.lms; self.gmstats = Matrix; self.gmgraph=False
        elif isinstance(Matrix, Graph): self.gmgraph = True ; self.gmstats = False
        else: self.lms = Matrix ; self.gmstats = False; self.gmgraph=False

        self.gfilter   = gfilter
        self.threshold = threshold
        self.unweighted= unweighted
        self.membership= None
        self.prefix    = prefix
        self.rand      = rand

        #check if graph have been done before if so load it is not execute buildgraph
        if isinstance(Matrix, ndarray) and not self.gmgraph:
            if self.prefix:
                if not isfile(str(self.prefix) + '.igraph.pickl'): self.Build_igraph()
                else: self.g = pickle.load(open(prefix + '.igraph.pickl'))
        elif isinstance(Matrix,Graph):
            self.g = Matrix

    def Build_igraph(self):
        '''
        Build a graph, using igraph library. It will return it, and store it as an attribute (self.g)

        :returns: :class:: 'igraph.Graph' 
        '''
        # create a multiplier if the weights are not high enough for resolution
        if (self.lms.max()/100000.) <= 1e-4: multiplier = 100000
        else: multiplier=1

        g = Graph(len(self.lms))
        for i in range(len(g.vs)):
            g.vs[i]['label'] = i

        for i in range(len(self.lms)):
            for j in range(i+1,len(self.lms)):
                #Assign edges if value is over the threshold
                if self.lms[i][j] and self.lms[i][j] >= self.threshold:
                    if self.gfilter:
                        #Only add edges if not filtered out
                        if ((i,j) in self.gfilter) or ((str(i),str(j)) in self.gfilter):
                            g.add_edge(i,j)
                            if not self.unweighted:##if unweighted, don't include weight
                                g.es[len(g.es)-1]['wts'] = int(self.lms[i][j]*multiplier)
                    else:
                        g.add_edge(i,j)
                        if not self.unweighted:##if unweighted, don't include weight
                            g.es[len(g.es)-1]['wts'] = int(self.lms[i][j]*multiplier)
        self.g = g
        g.write_pickle(fname=self.prefix + '.igraph.pickl')
        g.write(self.prefix+'.edges')		
        return g

    def Get_StructProps(self,overall=False):
        ''' 
        Get the structural properties in the graph

        :param overall: Calculate the centralities on the full graph, as opposed as by modules ( Default behaviour ).
        :type overall: boolean
        '''
        if self.unweighted: weights = None
        else: weights = self.g.es['wts']

        if overall:
            self.betweeness       = self.g.betweenness(weights)
            self.edge_betweenness = self.g.edge_betweenness(weights)
            self.degree           = self.g.degree(weights)
            self.closeness		  = self.g.closeness(weights)
            self.eigen 			  = self.g.eigenvector_centrality(weights)
            self.shell_index	  = self.g.coreness()
        else:
            mem = self.membership
            evcen,btcen,clcen,degree,edge_bt,shell=[],[],[],[],[],[]

            for e in mem:
                sg = g.subgraph(e)
                evcen.append(sg.eigenvector_centrality(weights))
                btcen.append(sg.betweenness(weights=sg.es['wts']))
                clcen.append(sg.closeness(weights))
                degree.append(sg.degree(weights))				
                edge_bt.append(sg.edge_betweenness(weights))
                shell.append(sg.coreness())

        self.betweeness       = btcen
        self.edge_betweenness = edge_bt
        self.degree           = degree
        self.closeness		  = clcen
        self.eigen 			  = evcen
        self.shell_index	  = shell


    def Graph_Cluster(self, method = 'fastgreedy',**kwargs):
        '''
        Clustering by components comstGreed, using igraph library.

        :param method: method in igraph ofr community detection. It can be fastgreedy, infomap leading_eigenvector_naive,leading_eigenvector,label_propagation, multilevel, optimal_modularity, edge_betweenness, spinglass, walktrap. For details see igraph documentation.
        :type method: string.
        :param kwargs: other arguments passed to the igraph methods
        '''
        self.ClusterMethod = method
        memgreed = [0]*self.lms.shape[0]
        index = 0
        #get independent componets of the graph
        comp = self.g.components()
        print "There are %d components in the network."%(len(comp))
        # Check if only singletons
        if len(comp) == len(self.lms):
            Warning('Only singletons in the dataset')
            mem = range(len(comp))
            for e in range(len(memgreed)):
                memgreed[e] = e
        else:
            # loop over each component and perform modularity optimization given the method provided
            for cp in range(len(comp)):
                cpn = comp.subgraph(cp)
                if len(cpn.vs) > 1:
                    try: mem = getattr(cpn,'community_'+method)(cpn.es['wts'],**kwargs).as_clustering()
                    except: mem = getattr(cpn,'community_'+method)(cpn.es['wts'],**kwargs)
                    for i in mem:
                        for j in i:
                            memgreed[cpn.vs[j]['label']] = index
                        index += 1
                else:
                    memgreed[comp[cp][0]] = index
                    index += 1
        self.mem = memgreed
        self.vertexclus = VertexClustering(self.g,self.mem)
        self.membership = self.vertexclus.membership
        try: self.modularityScore = self.g.modularity(self.membership, self.g.es['wts'])
        except: self.modularityScore = self.g.modularity(self.membership)

        print "%d cluster(s) found."%(len(set(self.membership)))
        print "Modularity score: %f"%(self.modularityScore)		


    def Cluster2File(self):
        ''' Write cluster to a file and rename the cluster with PDB friendly characters if possible (this is specific use)'''
        chains = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','Y','Z',
                  'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','y','z',
                  '0','1','2','3','4','5','6','7','8','9']
        line=''
        if isinstance(self.membership ,str): memb = self.membership.strip().split()
        if not len(set(memb)) > len(chains):
            for i in memb:
                if isinstance(i,str) and i.isdigit():
                    line += chains[int(i)]+' '
                elif isinstance(i,str) and not i.isdigit():
                    line += i + ' '
        else:
            for i in memb:
                line += str(i)+' '
        with open(self.prefix+'.graphcluster','w') as fout:
            fout.write(line)
        print 'Membership vector found:'
        print line
        print 'Membership vector written to %s'%(self.prefix+'.graphcluster')
        self.membership = line.strip().split()

    def Identify_Singletons(self,method='fastgreedy'):
        ''' Given a membership vector identify singletons'''
        if not self.membership: self.Graph_Cluster(method=method)
        nc=''
        d = Counter(self.membership)
        for e in self.membership:
            if d[e] == 1:
                nc+= '? '
            else:
                nc+= '%s '%(str(e))
        self.membership = nc

    def LDAmerge(self,which='lms',ellipses=2.5,dimensions=1):
        '''
        Perform an LDA analisis and merge all classes which 95% confidence ellipses collide.

        :param which: Either 'gm' or 'lms'. To perform the LDA in the full matrix or only in the correlation matrix.
        :type which: string
        :param ellipses: A value representing the estandard deviations for confidence ellipses. By default is set to 2.5 (95% confidence ellipses)
        :type ellipses: float
        :param dimensions: dimensions of the matrix
        :type dimensions: integer
        '''
        if self.gmstats:	
            self.gmstats.LDA(self.membership, which=which, ellipses=ellipses)
            fit = self.gmstats.fit
        else: 
            stats = GMstats(self.prefix, self.lms, dimensions=dimensions, 
                            sample_size=None,rand=self.rand)
            stats.LDA(self.membership, which=which, ellipses=ellipses)
            fit = stats.fit

        Ellipses = getEllipses(fit, self.membership, std=ellipses)
        merges=[]
        #Check for collisions between classes with 95% conficence ellipses
        K=Ellipses.keys()
        for i in range(len(Ellipses)):
            for j in range(len(Ellipses)):
                if i > j:
                    if EllipseOverlap(Ellipses[K[i]],Ellipses[K[j]]):
                        merges.append((K[i],K[j]))

        #merge unsupported clusters
        newcl = equiclus(merges)
        self.newmem = rename_clusters(newcl, self.membership)	





###############################################################################################
class SupportClustering:
    ''' 
    This class provides ways to provide statistical support for a given clustering scheme. It is based in
    testing if the correlation between groups is significatly different than  between groups.

    :param prefix: a prefix for the output.
    :type prefix: string
    :param data: a 2D numpy array with the data from which the clustering was inferred.
    :type data: :class `numpy.ndarray`
    :param membership: a list equal to the second dimension of data (i.e. data.shape[1]), corresponding to the clustering scheme being tested.
    :type membership: list
    :param dimensions: Number of dimensions in your data matrix. If your matrix is correlation or related, dimesions should be one.
    :type dimensions: integer
    :param permutations: Number of permutations to perform the permutation t-test.
    :type permutations: integer
    :param confval: confidence value for the test (1 - alpha).
    :type confval: float
    :param threshold: Value to filter out values of the correlation. If set to none, no threshold filtering will be done.
    :type threshold: float or None
    :param rand: wether to test if your data is better than random. Default is true
    :type rand: boolean
    '''	
    def __init__(self,prefix, data, membership, dimensions, permutations, confval=0.95, 
                 threshold=0.0, rand=True):
        self.prefix  = prefix
        if isinstance(data,GMdata): 
            self.data   = data.data
            self.GMdata = data
        else: 
            self.data   = data
            self.GMdata = None
        self.confval = confval
        if self.data.shape[0] == self.data.shape[1]: self.lms = self.data
        else: self.DealDimensions()
        self.dim  = dimensions
        self.perm = permutations
        self.thres= threshold
        self.mem  = membership
        self.rand = rand

    def DealDimensions(self):
        ''' 
        If the data has more than one dimension (in the cartesian sense or the origin of the data),
        split it, compute correlation of each dimension and then aggregate it using euclidean distance
        of the coefficients of determination. It assumes that the dimensions are interleaved.
        This correlation does not have a significance testing, use caution.
        It is reccomended to use GMstats class before using this class.
        '''
        dims=[]
        Rs=0
        for i in xrange(self.dim):
            dims.append(range(i,len(self.data.shape[1]),self.dim))

        for d in dims:
            C = corrcoef(self.data[:,d], rowvar=0)
            Rs+= C**2
        self.lms = sqrt(Rs)		

    def subsetTest(self, k1,k2):
        '''
        Given a series of indices, split the data into intra and inter correlation
        and test for equality

        :param subsetdata:
        :param k1: Name of one of the clusters being compared
        :type k1: string or integer
        :param k2: Name of the other clusters being compared
        :type k2: string or integer
        '''
        # if threshold not required (i.e. threshold false or none) set to a really negative number
        if self.thres: threshold = self.thres
        elif self.thres != 0.0: threshold = -1e300
        else: threshold=0.0

        # get the indices of each group from lms
        indA = where(array(self.mem) == k1)
        indB = where(array(self.mem) == k2)
        # get the elements corresponding to intragroups (A and B) and intergroups (AB)
        # only the upper triangle
        A    = self.lms[indA[0],:][:,indA[0]][triu_indices(len(indA[0]))]
        B    = self.lms[indB[0],:][:,indB[0]][triu_indices(len(indB[0]))]
        AB   = self.lms[indA[0],:].flatten()
        # select cases above the threshold
        A    =  A[where(A  > threshold)]
        B    =  B[where(B  > threshold)]
        AB   = AB[where(AB > threshold)]
        if not self.rand:
            # test if random cases are significant
            RA   = randm(A.shape[0])
            RAB  = randm(AB.shape[0])
            pvalR= twotPermutation(RA,RAB,self.perm)
            if pvalR < (1-self.confval):
                print 'Not enough data to discern %s vs %s from random. Setting to insignificant'%(k1,k2)
                pvalA=1
                pvalB=1
        else:
            if not isbetterthanrandom(self.dim, self.data):
                print 'Not enough data to discern %s vs %s from random. Setting to insignificant'%(k1,k2)
                pvalA=1
                pvalB=1                
            else:
                # perform the permutation t-test
                pvalA= twotPermutation(A, AB,self.perm)
                pvalB= twotPermutation(B, AB,self.perm)
            # store pvalues in their corresponding containers
            self.pvals.extend([pvalA,pvalB])
            self.cl['%svs%s_%s'%(k1,k1,k2)]= pvalA
            self.cl['%svs%s_%s'%(k2,k1,k2)]= pvalB

    def get_cluster_indices(self):
        '''
        Returns a dictionary with cluster indices
        '''
        if isinstance(self.mem ,str): self.mem = self.mem.strip().split()
        s = set(self.mem)
        d={}
        # get cluster indices
        for l in s:
            d[l]=[]
            for e in range(len(self.mem)):
                if self.mem[e] == l:
                    d[l].append(e)
        return d

    def filter_singletons(self,D):
        '''
        Giving a dictionary with cluster indices, return whichones are singletons

        :param D: a dictionary with cluster indices
        :type D: dictionary
        :returns: a dictionary with the indices of unconnected nodes
        '''
        singles={}
        if isinstance(self.mem,str): self.mem = self.mem.strip().split()
        C = Counter(self.mem)
        if '?' in C:
            # singletons already in membership vector
            Warning('Singletons might be already filtered. Check your results carefully')

        for i in C:
            # if the count of the class is 1, is a singleton
            if (C[i] == 1) or (i == '?'):
                singles[i]='?'
        #Create a new membership vector replacing singletons by ?
        newm=[]
        for m in self.mem:
            if m in singles.keys(): newm.append('?')
            else: newm.append(m)


        return singles, newm

    def AreModneighbours(self,A,indA,indB):
        ''' 
        loop over the adjacency matrix to find if indA and indB are in the neiborhood

        :param A: a list of tupples of related entries
        :type A: list
        :param indA: indices of grup A
        :type indA: list
        :param indB: indices of grup B
        :type indB: list
        :returns: boolean of whether or not indA and indB are neighbours
        '''
        ans = False
        for i in indA:
            for j in indB:
                if A[i][j] == 1:
                    ans = True
        return ans	

    def FDR_correction(self):
        '''
        Compute the False Discovery Rate correction for the critical value

        '''

        #sort and reverse the list for FDR correction
        self.pvals.sort()
        self.pvals.reverse()
        if len(self.pvals)<=2:
            FDR=1-self.confval
        else:
            for el in self.pvals:
                if float(el) <= ((self.pvals.index(el)+1)/len(self.pvals))*(1-self.confval):
                    FDR=((self.pvals.index(el)+1.0)/len(self.pvals))*(1-self.confval)
                    break
                else:
                    continue
            try:
                FDR
            except:
                FDR=((1-self.confval)*(len(self.pvals)-1))/(2*len(self.pvals))	

        self.FDR= FDR

    def FDRc_sigtest(self):
        '''
        Perform the logical significance test using FDR corrected critical value. Returns a binary
        dictionary of the comparisons
        '''
        self.scl={}
        # loop over cl dictionary to write binary permt output into a dictionary
        for k, v in self.cl.iteritems():
            if float(v) <= self.FDR:
                self.scl[k] = True
            else:
                self.scl[k] = False	


    def write_permt(self, newm,count=''):
        '''
        Write and print the output of the permutation test, the new membership vector, and do some 
        cleanup
        '''
        # Rename the previous graphcluster (the original of modularity)
        #rename(self.prefix+'.graphcluster',self.prefix+'.%sold_graphcluster'%(str(count)))

        # Some lines of the output
        line0='######## Significance test of Clustering  ########\n'
        line1='#'*8+' Test of Clustering (per_t-test)  ' + '#'*8 + '\n'
        n=('#'*50)+'\n'
        print(n+line1+n)		
        FDRc='FDR-corrected Critical value = '
        print(FDRc+str(self.FDR)+'\n')
        theader='Comparison\tp-value\tSignificant?\n'
        print(theader)		
        # Open the new graphcluster file and the permt output file
        fname=self.prefix+'.permt'
        with open(fname,'w') as fout:
            fout.write(n+line1+n)
            fout.write(FDRc+str(self.FDR)+'\n')
            fout.write(theader)
            # Write the comparisons
            if self.cl:
                for k in self.cl:
                    print k + '\t' + str(self.cl[k]) + '\t' + str(self.scl[k])
                    fout.write(k + '\t' + str(self.cl[k]) + '\t' + str(self.scl[k])+'\n')
            else:
                cline='No adjacent clusters in the graph. Pvalue < 10^-4 or singletons.'
                print cline
                fout.write(cline)
            print(n)
            fout.write(n)

        # Print and write the new membership vector
        if isinstance(newm,str):  newm= newm.strip().split()
        clustn = len(set(newm))
        if '?' in newm:
            clustn = clustn-1
        print str(clustn) + ' significant clusters'
        print 'New membership vector:'
        newvec=''
        for y in newm:
            newvec+= str(y)+' '
        print newvec
        with open(self.prefix+'.graphcluster','w') as clus:
            clus.write(newvec)	

        return newvec	

    def permt(self,filterin,count=''):
        '''
        Perform the permutation test over all pairs of classes in the membership vector provided

        :param filterin: a list of tuples of pairs of variable to be inlcuded (i.e. adjacency list if mem was provided by a graph)
        :type filterin: list
        :param count: Step in which this function has been called. Is for Iterative usage.
        :type count: string or integer
        '''
        #provide containers for pvalues
        self.pvals=[]
        self.cl ={}
        D = self.get_cluster_indices()
        singles, m = self.filter_singletons(D)		
        if len(singles) == len(self.mem):
            print('Only singletons in the dataset')
            print('Bye Bye')
            pass
        elif len(D) == 1:
            print 'Only one cluster. Bye, Bye'
            pass
        # print progress to screen and read in the membership vector
        print(str(len(D)*(len(D)-1))+' comparisons')
        print('Current membership vector:')
        print ' '.join([str(x) for x in m])
        print('Progress:')		
        E = deepcopy(D)
        for i in E.keys():
            if i in singles:
                del E[i]

        # if '?' in E.keys():
        # 	del E['?']
        keys=[]	
        neighbours=[]
        for k,v in E.iteritems():
            keys.append(k)
            indA=[int(x) for x in E[k]]
            for ke, va in E.iteritems():
                if not ke in keys:
                    indB=[int(x) for x in E[ke]]
                    if filterin:
                        if self.AreModneighbours(filterin,indA,indB):
                            self.subsetTest(k, ke)
                            neighbours.append((k,ke))
                        else:
                            continue
                    else:
                        self.subsetTest(k, ke)
        # FDR correction
        FDR = self.FDR_correction()
        # Perform the logical comparisons using FDR-corrected critical value
        self.FDRc_sigtest()
        '''
		popkeys={}
		for k in D.iterkeys():[0, 0, 2, 0, 4, 0, 5, 0, 0, 6, 0, 7, 8, 9]
			if not k == '?':
				popkeys[k]=k'''
        # Merge non-significant and reciprocal clusters
        # Get the merging events 
        merg=[]
        if not filterin:
            ncls=[]
            for j in D.iterkeys():
                for k in D.iterkeys():
                    if not (j == '?' and not k == '?') and not (j in singles or k in singles):
                        try:
                            if ord(j) > ord(k):
                                ncls.append((j,k))
                        except:
                            if j > k:
                                ncls.append((j,k))
        else:
            ncls = neighbours
        for l in ncls:
            i,e = l
            try:
                if self.scl['%svs%s_%s'%(i,i,e)] == self.scl['%svs%s_%s'%(e,i,e)] == False:
                    merg.append((i,e))		
            except:
                if self.scl['%svs%s_%s'%(i,e,i)] == self.scl['%svs%s_%s'%(e,e,i)] == False:
                    merg.append((i,e))						
        # List of sets of equivalent clusters
        newcl = equiclus(merg)
        # Rename clusters using the smallest label in each set
        newm = rename_clusters(newcl, self.mem)		
        #open Outfile and write result
        newvec = self.write_permt(newm, count=count)
        self.mem = newm
        return newm, newvec

    def VectorToEdgeList(self,v):
        ''' 
        Convert a membership vector to a list of edges.
        The membership vector must be a list or space separated string.

        :param v: a membership vector in list form
        :type v: list
        '''
        if isinstance(v,str): v.strip().split()

        # Output vector
        out = []

        # Process edges only once (no symmetry)
        for i in range(len(v)):
            for j in range(i,len(v)):
                if v[i] == v[j]:
                    out.append('%s.%s'%(i,j))

        # return edge list
        return set(out)

    def iterative_permt(self,filterin):
        '''
        Perform a permutation test iteratively, until a stable membership vector is reached. In each iteration
        a permutation test is performed between the clusters, and a merge event will happen if no evidence of
        difference is found.

        :param filterin: a list of tuples of pairs of variable to be included (i.e. adjacency list if mem was provided by a graph)
        :type filterin: list
        '''
        memvec=[' '.join([str(x) for x in (self.mem)])]

        if not set(self.mem) == 1:
            count=0
            while count <= 20:
                count+=1
                it= 'Performing iteration number %d'%(count)
                print '\n\n%s\n%s\n%s\n'%('*'*(len(it)),it,'*'*(len(it)))
                newm, newvec = self.permt(filterin,count=count)
                memvec.append(newm)
                if len(memvec) >= 2:
                    old = self.VectorToEdgeList(memvec[-2])
                    new = self.VectorToEdgeList(memvec[-1])
                    fsc = Fscore(Specificity(old,new), Sensitivity(old,new))
                    if fsc >= 0.9999:
                        break
            self.mem = newm
            return newm, newvec
        else:
            print 'Only one cluster, avoiding iterations'

    def if_bootfile(self, boot=100):
        '''
        if another bootsrap intance has been called and crashed, this function will finished
        the remaining and / or compute the agreement

        :param boot: Number of bootstrap replicates to be performed
        :type boot: integer
        '''
        vectors = []
        if isfile(self.prefix+'.bootstrap'):
            print 'Previous bootstrap run found. Using %s file'%(self.prefix+'.bootstrap')
            with open(self.prefix+'.bootstrap') as f:
                for e in f:
                    if e == '' or e == '\n':
                        continue
                    else:
                        m = e.split('\t')[1].strip('[').strip(']').replace(' ','').replace('"','').replace("'","").split(',')
                        vectors.append(m)                
            dif = int(boot) - len(vectors)
            if dif > 0:
                # reopen boot file and keep going
                F = open(self.prefix+'.bootstrap','a')
                return F, xrange(len(vectors),boot), vectors
            else: return None, [], vectors
        else:
            return open(self.prefix+'.bootstrap','a'), xrange(len(vectors),boot), vectors

    def bootstrap(self,boot=0,contacts=[],unweighted=False,graphmethod='fastgreedy', lda=False,
                  iterative=True,Matrix=False,**kwargs):
        '''	
        Execute the bootstrap inferfence. This boostrap resample obsevations (rows) in the data

        :param boot: Number of bootstrap replicates to be performed
        :type boot: integer
        :param contacts: filter out non-contact interactions. Contacts passed as list of tuples
        :type contacts: list
        :param unweighted: create an unweighted graph as opposed to a weighted (correlation; default) one.
        :type unweighted: boolean
        :param graphmethod: method in igraph ofr community detection. It can be fastgreedy, infomap leading_eigenvector_naive,leading_eigenvector,label_propagation, multilevel, optimal_modularity, edge_betweenness, spinglass, walktrap. For details see igraph documentation.
        :type method: string.
        :param lda:	Use LDA to premerge the community detection clusters
        :type lda: boolean
        :param kwargs: parameter and arguments for GMstats class in the method Compute_correlations
        :param Matrix: user defined matrix. By default a file will be read, based on :class GMdata.
        :type Matrix: :class `numpy.array`
        '''
        bout, Range, vectors = self.if_bootfile(boot=boot)
        if not Range: self.bootvectors = vectors
        if not self.GMdata: 
            if not isinstance(Matrix,bool): self.GMdata = GMdata(self.prefix,dimension=self.dim,Matrix=Matrix)
            else: self.GMdata = GMdata(self.prefix,dimension=self.dim)
        #backup data
        #F = glob(prefix+'*')
        #for f in F:
        #	# get the extension and filename
        #	ext = f[f.rfind('.'):]
        #	name = f[:f.find(ext)]
        #	nf=name+'_original'+ext
        #	# rename the file
        #	rename(f,nf)

        for i in Range:
            bootprefix = self.prefix+'_boot%d'%i
            l='\n#  BOOTSTRAP REPLICATE %d  #\n'%(i)
            print '#'*(len(l)-2),l,'#'*(len(l)-2)
            D = self.GMdata.bootstrap_replicate()
            BootStats = GMstats(self.prefix, D, dimensions=self.dim, sample_size=D.shape[1])
            BootStats.Compute_correlations(**kwargs)
            if 'additive' in kwargs: BootStats.agglomerare_additive()
            else: BootStats.agglomerare_mean()

            BootGraph = GMgraph(bootprefix, BootStats, unweighted=unweighted, gfilter=contacts,
                                threshold= kwargs['threshold'])
            if contacts:
                filterin = GMgr.g.get_adjacency() # get the adjacency matrix to filter permt
            else:
                filterin = []				

            BootGraph.Identify_Singletons(method=graphmethod)#Graph_Cluster(method=graphmethod)
            mem = BootGraph.membership
            if lda:
                if not contacts:
                    print 'WARNING!! Using Linear discriminant pre-filtering without contacts '\
                          'will underestimate the number of modules!! Use the non lda filtered option instead.'
                print 'Membership vector before LDA (singletons are labelled as ?):'
                print mem
                BootGraph.LDAmerge()
                mem = BootGraph.newmem
                print 'Membership vector after LDA:'
                print mem

            BootSigClust = SupportClustering(bootprefix, BootStats.lms, mem, self.dim, self.perm, confval=self.confval, 
                                             threshold=self.thres)
            if iterative:
                BootSigClust.iterative_permt(filterin)
            elif self.perm:
                newm, _ = BootSigClust.permt(filterin)
                BootSigClust.mem = newm
            bout.write('Boot %d#\t%s\n'%(i,BootSigClust.mem))
            vectors.append(BootSigClust.mem)
            self.bootvectors = vectors

        #edges = [self.VectorToEdgeList(v) for v in vectors]
        self.bipartition = self.bipartition_agreement(self.prefix)
        self.WriteBoot()

    def bipartition_agreement(self, prefix):
        ''' 
        Calculate the local bipartition agreement scores
        '''
        vectors = self.bootvectors
        if isinstance(self.mem,str): refvec = self.mem.strip().split()
        else: refvec = self.mem
        clusters = sorted(list(set(refvec)))
        refbipartitions = {} ## dictionary containing each cluster's bipartition
        for clus in clusters:
            refbipartitions[clus] = ''
            for i in range(len(refvec)):
                if refvec[i] == clus:
                    refbipartitions[clus] += '1' ## 1 for dot
                else:
                    refbipartitions[clus] += '0' ## 0 for dash

        # Enumerate all the bipartitions in the replicate set
        repbipartitions = {}
        for i in range(len(vectors)):
            if isinstance(vectors[i],str): repvec = vectors[i].strip().split()
            else: repvec=vectors[i]
            cluses = sorted(list(set(repvec)))
            for clus in cluses:
                partition = ''
                for i in range(len(repvec)):
                    if repvec[i] == clus:
                        partition += '1'
                    else:
                        partition += '0'
                if partition not in repbipartitions:
                    repbipartitions[partition] = 1
                else:
                    repbipartitions[partition] += 1

        # Calculate the score
        FOUT = open(prefix+'.scores.bootstrap','w')
        bipartitions = {}
        total = sum(repbipartitions.values()) ## total number of bipartitions in the replicate set
        for clus in refbipartitions:
            agreement = 0
            for p in repbipartitions:
                if self.BipartitionAgree(refbipartitions[clus], p):
                    agreement += repbipartitions[p]
            bipartitions[clus] = round(float(agreement)/total, 3)

        for k,v in bipartitions.iteritems():
            FOUT.write(k+'\t%f\n'%(v))

        FOUT.close()
        return bipartitions	


    def BipartitionAgree(self, a, b):
        '''
        Return whether 2 strings of bipartitions agree or conflict. The strings must consist of
        1 and 0 only

        :params a,b: binary bipartition strings to be compared for agreements.
        :type a,b: strings
        '''
        if len(a) != len(b):
            raise Exception(a + ' and ' + b + ' are not of the same length to compare bipartitions.')

        else:
            pairs = []
            for i in range(len(a)):
                if (a[i], b[i]) not in pairs:
                    pairs.append((a[i], b[i]))
            return len(pairs) < 4	

    def WriteBoot(self):
        '''
        Write Bootstrap results to screen and file
        '''
        line = '\nBootstrap results\n'
        line = '*'*len(line)+line+'*'*len(line)+'\n'
        for i in sorted(self.bipartition.keys()):
            line += '%s = %f\n'%(i,self.bipartition[i])
        with open(self.prefix+'.bipartitions','w') as F: F.write(line)
        print(line)

###############################################################################################
class CreateHierarchy:
    def __init__(self, inst):
        '''
        Helper class to ModuleStructure to create the hierarchical structure
        based on an instance of the class
        :param inst: instance of the class ModuleStructure
        :type :class `Moduler.ModuleStructure`
        '''
        self.lis = []
        self.inst= inst
        self.populate_hierarchy()

    def check_child(self): 
        return len(self.inst.child)

    def recurse(self,inst,lis):
        #inst = self.inst
        ma = max(inst.mem)
        lis.append((inst.level,inst.mem,inst.indices))
        if inst.child:
            for i in xrange(self.check_child()):
                ch = inst.child[i]
                lis.append((ch.level,ch.mem,ch.indices))
                self.recurse(inst.child[i],lis)

    def fix_mem(self,highestprev,mem):
        '''TO DO'''
        pass

    def lowest_level(self):
        inst = self.inst
        lev = []
        self.recurse(inst,lev)
        return max([x[0] for x in lev]), lev

    def fill_hierarchy(self):
        for i in xrange(self.low-1):
            h = self.hierarchy[i]
            l = self.hierarchy[i+1]
            if any(equal(l,None)):
                ind = nonzero(equal(l,None))[0]
                l[ind]=h[ind]

    def populate_hierarchy(self):
        inst = self.inst
        hierarchy = []
        lowe, tup = self.lowest_level()
        self.low  = lowe+1
        lenvec    = len(inst.mem)
        for i in xrange(self.low):
            hierarchy.append(array([None]*lenvec))
        for x in tup:
            hierarchy[x[0]][x[2]]=x[1]
        self.hierarchy = hierarchy
        self.fill_hierarchy()

class ModuleStructure:
    ''' 
    Get structure for hierarchical modules

    :param graph: a graph instance after the first run of modularity
    :type graph: :class `Graph`
    :param matrix: The matrix used in the construction of the graph
    :type matrix: :class `numpy.ndarray`
    :param membership: A membership vector for the current structure
    :param hierarchy: the array of the hierachichal module structure
    :type hierarchy: None or :class `numpy.ndarray`
    :param parent: Pointer to the parent class of the current structure
    :param level: level in the hierarchy
    :type level: int
    :type: :class `ModuleStructure` or None
    :param method: the clustering method as passed to igraph
    :type method: str
    :param modulelabel: The label of the module being analized
    :type modulelabel: None or int
    :param lda: perform LDA pre-merge in each instance of the hierarchy
    :type lda: boolean
    '''	
    def __init__(self,graph,matrix,membership,level=0,parent=None, dim=3,
                 modulelabel=None,method='fastgreedy',lda=False):
        self.dim = dim
        self.parent = parent
        self.child = []
        self.graph = graph
        self.lms = matrix
        self.lda = lda
        if modulelabel:
            self.mem = [str(modulelabel)+str(x) for x in membership]
        else: 
            self.mem = membership
        self.Method    = method
        self.level     = level		

        if self.parent == None: 
            self.indices = arange(len(self.mem))
        else: 
            self.indices = self.parent.indices[where(modulelabel == array(self.parent.mem))[0]]

        # Define attribute module
        if isinstance(self.mem, str):
            self.graph.vs['module'] = self.mem.strip().split()
        else:
            self.graph.vs['module'] = self.mem
        # populate
        self.populateModules()

    def subsetMatrix(self, subgraph):
        '''
        Given an original matrix and a subgraph, slice the submatrix 
        corresponding to that subgraph
        :param subgraph: a subgraph being analysed
        :type subgraph: :class `Graph.subgraph`
        '''
        ind = subgraph.vs['label']
        submatrix = self.lms[ind,:][:,ind]
        return submatrix

    def populateModules(self):
        ''' Populate hirarchical module structure'''
        if len(set(self.mem)) <= 1: return 

        for gr in set(self.mem):
            ind = where(gr == array(self.mem))[0]
            sg  = self.graph.vs.select(module=gr).subgraph()
            mat = self.subsetMatrix(sg)
            graph,matrix,modules,parent,method = self.modularity(gr,sg, mat,lda=self.lda)
            Ms  = ModuleStructure(graph,matrix,modules,self.level+1,parent,self.dim, gr,
                                  method,lda=self.lda)
            self.child.append(Ms)
            Ms.populateModules()

    def modularity(self,prefix,graph,matrix,lda=False):
        '''
        Compute the modularity for a given graph/subgraph, with a given matrix

        :param graph: A graph or subgraph to which apply modularity inference
        :type graph:  :class `Graph` or :class `Graph.subgraph`
        :param matrix: Data matrix source of graph
        :type matrix: :class `numpy.ndarray`
        '''
        graph.vs['label'] = range(len(self.graph.vs))
        GMg    = GMgraph(None, graph)
        GMg.lms= matrix
        GMg.Identify_Singletons(self.Method)
        if lda:
            if self.dim > 1: 
                typ = 'gm'
            else:
                typ = 'lms'
            GMg.LDAmerge(which=typ, dimensions=self.dim)
        SigClust = SupportClustering(str(prefix), matrix, GMg.membership,self.dim,999)
        SigClust.iterative_permt(GMg.g.get_adjacency())
        modules = SigClust.mem		
        return graph,matrix,modules,self,self.Method

###############################################################################################


# Auxiliary methods
def twotPermutation(x1,x2,perm):
    '''
    This function computes the p-value for the two sample t-test using a permutation test.
    This is a translation from the DAAG Package function with the same name.
    :param x1: First array (sample) to be compared
    :type x1: :class `numpy.ndarray`
    :param x2: Second array (sample) to be compared
    :type x2: :class `numpy.ndarray`
    :param nsim: number of simulations to run ro compute the ovalue
    :type nsim: integer
    '''	
    nsim=perm
    n1 = len(x1)
    n2 = len(x2)
    n  = n1 + n2
    x  = concatenate((x1,x2))
    A  = array(xrange(len(x)))
    s  = set(A)
    dbar = float(mean(x2) - mean(x1))
    z = zeros(nsim)
    for i in xrange(nsim):
        mn = sample(xrange(n), n2)
        w = x[mn]
        wo= x[list(s.difference(set(A[mn])))]
        z[i] = mean(w) - mean(wo)
    pval = (sum(z >= abs(dbar)) + sum(z <= -abs(dbar)))/float(nsim)
    return pval

def isFloat(string):
    '''
    Auxiliary function to test if a string is a float

    :param str string: The string to be tested
    '''
    try:
        float(string)
        return True
    except:
        return False


def rename_clusters(newcl, mem):
    '''
    Rename clusters using the smallest label in each set. Returns the new
    membership vector

    :param newcl: a list with the new membership before renaming
    :type newcl: list
    :param mem: original membership vector
    :type mem: list
    :returns: A list with the renamed vector
    '''
    babel = {}
    for cluster in newcl:
        newname = min(cluster)
        for item in cluster:
            babel[item] = newname

    for i in range(len(mem)):
        if mem[i] in babel:
            mem[i] = babel[mem[i]]
    # Create the new membership vector
    newm = []
    for ite in mem:
        newm.append(ite)

    # Count occurrences
    C =  Counter(newm)

    newmm=[]
    # deal with singletons
    for i in newm:
        if C[i] == 1: newmm.append('?')
        else: newmm.append(i)

    return newmm

def equiclus(merge):
    '''
    Look for equivalent clusters, and return a list with sets of equivalent clusters

    :param merge: list of tuples with the pairs of classes to be merged
    :type merge: list
    :returns: cleaned membership vector as a list
    '''

    # List of sets of equivalent clusters
    newcl = []	
    for m in merge:
        m = set(m)
        merged = False
        # look for a newcl item which overlaps then merge
        for nc in range(len(newcl)):
            xnc = newcl[nc]
            if m.intersection(xnc):
                # Merge into xnc
                newcl[nc] = xnc.union(m)
                merged = True
                break
        if not merged:
            # New cluster to add to newcl
            newcl.append(m)

    return newcl

def ellipse(singlegroupx,singlegroupy,std=2.5):
    ''' 
    Create an ellipse given points with x coordinates in singlegroupx and singlegroupy

    :param singlegroupx: An array with the x coordinates of data for what the ellipse is to be built.
    :type singlegroupx: :class `numpy.array`
    :param singlegroupy: An array with the y coordinates of data for what the ellipse is to be built.
    :type singlegroupy: :class `numpy.array`
    :param std: The standard deviation of the confidence ellipse. By default is set to 2.5 (95% confidence ellipse)
    :type std: float
    :returns: a :class `matplotlib.patches.Ellipse` object
    '''
    covar     = cov(singlegroupx, singlegroupy)
    if isnan(covar).any():
        width = 0.01
        height= 0.01
        angle = 0
    else:
        lambda_, v = eig(covar)
        lambda_    = sqrt(lambda_)
        width      = lambda_[0]*std*2
        height     = lambda_[1]*std*2
        angle      = rad2deg(arccos(v[0, 0]))

    centerx    = mean(singlegroupx)
    centery    = mean(singlegroupy)		
    ell        = Ellipse(xy=(centerx,centery), width=width , height=height , angle=angle)

    return ell

def pointsInEllipse(Xs,Ys,ellipse):
    '''
    Tests which set of points are within the boundaries defined by the ellipse. The set of points
    are defined in two arrays Xs and Ys for the x-coordinates and y-coordinates respectively.

    :params Xs: x-coordinates of a set of points
    :type Xs: :class `numpy.array` or list
    :params Ys: y-coordinates of a set of points
    :type Ys: :class `numpy.array` or list
    :param ellipse: instance of method ellipse
    :type ellipse: :class `matplotlib.patches.Ellipse`
    :returns: Tuple of two list, points inside and outside of the ellipse
    '''
    inside = []
    outside= []
    for i in range(len(Xs)):
        point = (Xs[i],Ys[i])
        if ellipse.contains_point(point):
            inside.append(point)
        else:
            outside.append(point)
    return inside, outside

def getEllipses(fit, membership, std=2.5):
    '''
    Populate the ellipses dictionary with as many ellipses as components in membership.

    :param fit: An array normally computed by LDA or other method
    :type fit: :class `numpy.array`
    :param std: The standard deviation of the confidence ellipse. By default is set to 2.5 (95% confidence ellipse)
    :type std: float
    :param membership: A list corresponding to memberships of the entries in data.
    :type membership: list
    :returns: A dictionary with Ellipse objects with membership as keys.
    '''
    ellipses={}
    for mem in list(set(membership)):
        sub = fit[where(array(membership) == mem)]
        if len(sub.shape) < 2: sub = sub.reshape((sub.shape[0],1))
        Xs=sub[:,0]
        Ys=sub[:,1]
        ellipses[mem]=ellipse(Xs,Ys,std=std)
    return ellipses

def EllipseOverlap(ellipse1,ellipse2):
    ''' 
    Find collisions between ellipses

    :param ellipse1: Matplotlib ellipse
    :type ellipse1: :class `matplotlib.patches.Ellipse`
    :param ellipse2: Matplotlib ellipse
    :type ellipse2: :class `matplotlib.patches.Ellipse`
    :returns: Boolean of whether or not two ellipses collide
    '''
    pointsin2=ellipse2.properties()['verts']
    ins, out = pointsInEllipse(pointsin2[:,0], pointsin2[:,1], ellipse1)
    if any(ins): return True
    else: return False




def Specificity(ref, test):
    '''
    Compute specificity from two edgelists (list of edges between nodes/labes)

    :param ref: Reference edgelist 
    :type ref: list
    :param test: Test edgelist
    :type test: list
    '''
    # Size of intersection
    A = len(ref.intersection(test))

    return float(A) / len(test)

def Sensitivity(ref, test):
    '''
    Compute sensitivity from two edgelists (list of edges between nodes/labes)

    :param ref: Reference edgelist
    :type ref: list
    :param test: Test edgelist
    :type test: list
    '''
    # Size of intersection
    A = len(ref.intersection(test))

    return float(A) / len(ref)

def Fscore(sp, sn):
    ''' 
    Compute the F-score 

    :param sp: Specificity value
    :type sp: float
    :param sn: Sensitivity value
    :type sn: float
    '''
    return 2*sp*sn/(sp+sn)

def clus2list(membership,data):
    '''
    Giving a membership vector and data, split the data into cluster specific arrays.

    :param membership: a membership vector
    :type membership: list or string
    '''
    if isinstance(membership,str): m = membership.split()
    else: m = membership
    #dictionary to hold the array corresponding to each cluster
    darr = {}
    smem = set(m)
    for s in smem:
        ind = where(array(m) == s)[0]
        if not s == '?':	
            if isinstance(data,list):
                temp=[]
                for d in data:
                    temp.append(d[ind,:][:,ind][triu_indices(len(ind))])
                darr[s] = temp
            else:
                if data.shape[0] == data.shape[1]: darr[s]=data[ind,:][:,ind][triu_indices(len(ind))]
                else: darr[s]=data[:,ind]

    return darr

def isbetterthanrandom(dim,data):
    ''' 
    Extension of T-method (Roff et. al. 1999) of equality in matrices. Here, a random matrix of 
    the same dimensions is tested
    :param dim: number of dimensions of the input data
    :type dim: float
    :param data: The data input matrix
    :type data: :class `numpy.ndarray`
    :returns: a boolean of wether or not you data matrix correlation is better than random
    '''
    
    def loop(row,col):
        r1,r2 = corrcoef(randm(row,col)), corrcoef(randm(row,col))
        diff  = abs(subtract(r1, r2))
        Tr    = sum(diff)
        return Tr
    boolean=[]
    for d in xrange(dim):#len(self.data)):
        if dim != 1:
            onematrix = data[d]
        else:
            if len(data.shape)!= 2:
                onematrix = data[0]
            else:
                onematrix = data
        row,col= onematrix.shape
        real   = corrcoef(onematrix)
        Random = corrcoef(randm(row,col))
        To = sum(abs(subtract(real, Random)))
        #Ts = Parallel(n_jobs=1)(delayed(loop)(row,col) for i in xrange(4999))
        Ts=[]
        for i in xrange(4999): Ts.append(loop(row, col))
        Ts = array(Ts)
        count = len(where(To < Ts)[0])
        Pt= (1.+count)/5000
        boolean.append(Pt < 0.05)
        if dim==1:
            break
    return all(boolean)


# Moduler specific methods... execution
def _Master_of_ceremony(prefix, dimensions=3,contacts=False,additive=True,method='fisher',
                        absolutecov=False,confval=0.95,threshold=0.0,usecov=False,
                        writemat=False,overall=False,lda=False,perm=0,iterative=True,
                        power=0.8,mapmodules=False,morph=False,boot=0,
                        graphmethod='fastgreedy',hierarchy=True, rand=True):
    # Say hi!!!#######################################################################
    print('\nWelcome to MODULER version %s'%(__version__))
    print('A python script to explore modularity on coordinates data\n\n')
    l = ''
    # Print the chosen parameters and write it to a file #########################
    l+= 'Chosen parameters:\n'
    l+= 'Test of significance for correlation = %s\nPower analysis = %s\n'%(method,power)
    l+= 'Aglomerate dimensions as Euclidean distance = %s\n'%(additive)
    l+= 'Confidence Value = %f (to be use in both correlation and significance test if true)\n'%(confval)
    l+= 'Write matrices to file: %s\n'%(writemat)
    l+= 'Absolute value of correlation = %s\n'%(absolutecov)
    l+= 'Dimensions = %d\n'%(dimensions)
    if perm > 0:
        l+='Single test for significance of clusters = True with %d permutations\n'%(perm)
    else:
        l+='Single test for significance of clusters = False\n'
    l+= 'Iterative test for significance of clusters = %s\n'%(iterative)
    l+= 'Filtering out non contact interactions = %s\n'%(contacts)
    l+= 'Linear discriminants prefiltering = %s\n'%(lda)
    l+= 'Bootstrap : %d (if 0 no bootstrap is performed)\n'%(boot)
    l+= 'Optimization method for modularity in the graph: %s'%(graphmethod)
    #l+= 'Analyse  interlandmark distances (if False will use raw coordinates) = %s'%(options.dist)
    print l
    ou = open(prefix+'.parameters', 'w') 
    now = datetime.datetime.now()
    ou.write('file created on %s\n\n'%(str(now))+l)
    return ou

def main(prefix, dimensions=3,contacts=False,additive=True,method='fisher', confval=0.95, power=0.8, absolutecov=False, 
         usecov=False, graphmethod='fastgreedy', threshold=0.0,writemat=False,overall=False,lda=False,perm=0,iterative=True,
         mapmodules=False,morph=False, boot=0, mc=None, unweighted=False, hierarchy=False,rand=True ):
    ''' 
    Execute the code Moduler as __main__

    :param prefix: Prefix of your GM file and accompanying files. Also will be use for output files
    :type prefix: string
    :param dimensions: The number of cartesian dimensions to be analyzed. By default is set to 3.
    :type dimensions: integer
    :param contacts: Constrain the graph to only things that are in contact. If True it will look for a file with '.contact' extension.If the file cannot be found it will look for a PDB file named <prefix>.pdb, and will try to compute the contacts from :class PDBstructure. If the pdb cannot be found it will raise an exception.
    :type contacts: boolean
    :param additive: Use the mean of the correlation in each dimension instead of the euclidean distance to aglomerate the dimensions. The default behaviour is additive.
    :type additive: boolean
    :param method: Which method to use for correlation. By default it uses the fisher transformation. Available options are pearson, spearman and fisher.        
    :type method: string
    :param confval: Define the confidence level (1-alpha) for the correlation test. By default is 0.95. Can take any float value between 0 and 1.
    :type confval: float
    :param power: Perform a statistical power analysis with this value as the desired Power (1-type II error). by default 0.8 is used. Can be any float from 0 to 1
    :type power: float
    :param absolutecov: Whether or not to use absolute values of the correlation matrix
    :type absolutecov: boolean
    :param usecov: Use covariance instead of correlation
    :type usecov: boolean
    :param graphmethod: Method to seach for the partition of the graph. By default is fastgreedy. Refer to igraph documentation in community method for other options.
    :type graphmethod: string
    :param threshold: A float corresponding to the threshold to create a conection between nodes. This is very user dependendent and is set to 0 as default.
    :type threshold: float
    :param writemat: Write correlation/covariance matrices to file. By default is false. It can take a False, for the aggreatated matrix or cor for the correlation matrix.
    :type writemat: boolean
    :param overall: Calculate the centralities on the full graph, as opposed as by modules (Default behaviour).
    :type overall: boolean
    :param lda:	Use LDA to premerge the community detection clusters
    :type lda: boolean
    :param perm: number of permutation for the significance tests
    :type perm: integer
    :param iterative: whether to perform significance test among clusters iteratevely until estabilization of the cluster.
    :type iterative: boolean
    :param mapmodules: Map the Modules in a PDB provided. The PDB and chain in which to map, have to be provided comma separated.
    :type mapmodules: string
    :param morph: Use this if you want to estimate the modularity of morphological data. Basically it will create a dummy landmark file.
    :type morph: boolean
    :param boot: Compute bootstrap for sample reliability estimation
    :param mc: a file handle from Master_of_Ceremony mathod.
    :type mc: opened file
    :param unweighted: create an unweighted graph as opposed to a weighted (correlation; default) one.
    :type unweighted: boolean
    :param hierarchy: If hierarchical modules should be explored.
    :type hierarchy: boolean
    :param rand: Whether to test if data is better than random in the correlation test. default is True
    :type rand: boolean
    ''' 
    args = {'method':method, 'absolutecov':absolutecov, "confval":confval, 'usecov':usecov,
            'writemat':writemat, 'additive':additive, 'power':False, 'threshold': threshold}
    # instantiate Data #####################################################
    GMd = GMdata(prefix,dimension=dimensions,contacts=contacts)
    # Load Contacts ########################################################
    gfilter = GMd.contacts
    # do the stats
    GMs = GMstats(prefix, GMd.data, dimensions=dimensions,sample_size=GMd.sample_size,rand=rand)

    if GMs.lms == None:
        GMs.Compute_correlations(method=method, absolutecov=absolutecov, confval=confval,threshold=threshold, 
                                 usecov=usecov, writemat=writemat, additive=additive, power=power)
        if additive: GMs.agglomerare_additive()
        else: GMs.agglomerare_mean

    # get Edge assignment threshold and print it to screen #################
    if threshold == 'Auto': 
        GMs.GetAglomeratedThreshold()
    elif mc:
        t = 'Threshold mode: Custom = %f'%(threshold)
        mc.write(t)	
        print t

    # Build igraph ###################################################################
    GMgr = GMgraph(prefix, GMs, unweighted=unweighted, gfilter=gfilter, threshold=threshold)
    GMgr.Build_igraph()
    if contacts:
        filterin = GMgr.g.get_adjacency() # get the adjacency matrix to filter permt
    else:
        filterin = []

    # Clustering by components comstGreed ############################################
    GMgr.Identify_Singletons(method=graphmethod)
    #memgreed = GMgr.mem
    mem = GMgr.membership

    # write to cluster file ##########################################################
    GMgr.Cluster2File()

    # run LDA to premerge clusters
    if lda:
        if not contacts:
            print 'WARNING!! Using Linear discriminant pre-filtering without contacts '\
                  'will underestimate the number of modules!! Use the non lda filtered option instead.'
        print 'Membership vector before LDA (singletons are labelled as ?):'
        print mem
        GMgr.LDAmerge()
        mem = GMgr.newmem
        print 'Membership vector after LDA:'
        print mem
    # significance test of clustering: Outputs a new graphcluster
    SigClust = SupportClustering(prefix, GMs.lms, mem, dimensions, perm, confval=confval, threshold=threshold)

    if iterative:
        SigClust.iterative_permt(filterin)
    elif perm:
        newm, newvec = SigClust.permt(filterin)
        SigClust.mem = newm

    # Perform options.power analyses
    if power:
        propor = GMs.Clus_Power(mem)


    if boot:
        SigClust.bootstrap(boot=boot,contacts=filterin,unweighted=unweighted, graphmethod=graphmethod,**args)

    # Compute hierarchical structure
    if hierarchy:
        MS = ModuleStructure(GMgr.g, GMs.lms, SigClust.mem, method=graphmethod)
        H = CreateHierarchy(MS)
        savetxt('%s.hierarchy'%(prefix), H.hierarchy, fmt='%s')

    # Map modules and centralities #########################################################
    if mapmodules:
        if GMd.pdb.IsTrajectory():
            names = GMd.pdb.GetModelNames()
        else:
            names = GMd.pdb.GetChainNames()
        GMd.pdb.Map2Protein(prefix+'.mapped.pdb', SigClust.mem, names[0], prefix+'.fasta')


# End of definitions###########################################################################

# Aplication of the code ######################################################################
if __name__ == "__main__":
    # Command line input #############################################################
    opts = optparse.OptionParser(usage='%prog <prefix> [options]')
    opts.add_option('-o','--covariance',dest='usecov', action="store_true",
                    default=False, help='''Use covariance instead of correlation to build the graph. If this option is not 
	                provided the program will use correlation by default.''')
    opts.add_option('-a','--mean', dest='additive', action="store_false", default=True,
                    help='''Use the mean of the correlation in each dimension instead of the euclidean distance to aglomerate
	                the dimensions. The default behaviour is additive.''')
    opts.add_option('-t', '--threshold', dest='threshold', action='store',# type=float, 
                    default=0.0, help= '''Set the threshold for edge assingment to the value provided. Otherwise is set to 0. 
	                An extra option is Auto which will calculate the treshold based on the mean standard deviation of the 
	                agglomerated dimensions matrix.''')
    opts.add_option('-p','--power', dest='power', action='store', type='float', default=0.8,
                    help='''Perform a power analysis with this value as the desired Power (1-type II error). Default: 0.8''')
    opts.add_option('-m','--method', dest='method', default='fisher', 
                    help='''Test the significance of the correlation by the method provided(pearson, spearman, 
	                kendall or fisher). The latter is the fisher transformation of the pearson correlation. If no test
	                needed False should be provided.''')
    opts.add_option('-i', '--confidencelevel', dest='confval', action='store', type='float',
                    default=0.95, help='''Define the confidence level (1-alpha) for the correlation test. 
	                Default: 0.95.''')
    opts.add_option('-d','--dimensions', dest='dimensions',action='store', type='int', default=3,
                    help='''Set the dimensions of the shape. Default: 3.''')
    opts.add_option('-c','--contact',dest='contacts', action="store_true", default=False,
                    help='''Use the all-atom contact matrix when assigning edges. Default: False.''')
    opts.add_option('-u','--absolute',dest='absolutecov',action="store_true",default=False,
                    help= '''Use absolute values of the correlation matrix. Default: False.''')
    opts.add_option('-n', '--permtest',dest='perm', action='store', type='int',default=999,
                    help='''Test the significance of clustering with N permutations. When 0 is passed, no permutation
	                test is performed.''')
    opts.add_option('-I','--iterative',dest='iterative', action='store_false',default=True,
                    help= '''Iterative (until convergence) test the significance of clustering with permutations 
	                being passed through permtest. ''')
    opts.add_option('-w','--matrix',dest='writemat', default=False, 
                    help='''Write to file the matrix of the option chosen. It can be either cor (which will print
	                the correlation matrix for each dimension), or agg will write to a file the 
	                agglomerated dimensions matrix.''')
    #opts.add_option('-r','--RV',dest='rv',action="store_false",default=True,
    #               help='Do not Write the the Excouffier RV for the modules inferred.'\
    #              ' The p-values will be given using the Pearson type III approximation.')
    opts.add_option('-f','--fullcentrality',dest='overall',action="store",default=False,
                    help='''Use this if you want to calculate the centralities on the full graph, as opposed as by
		            modules ( Default behaviour ).''')
    opts.add_option('-B','--bootstrap',dest='boot',action='store', type='int',default=0,
                    help='''Compute bootstrap for sample reliability estimation.''')	
    opts.add_option('-l','--lda',dest='lda',action="store_true",default=False,
                    help='''Use LDA to premerge the community detection clusters''')
    opts.add_option('-G','--graph',dest='graphmethod',action="store",default='fastgreedy',
                    help='''Use one of igraph methods for community detection inference. By default is fastgreedy
	                but it can take: infomap, leading_eigenvector_naive, leading_eigenvector, label_propagation, 
	                multilevel, optimal_modularity, edge_betweenness, spinglass, walktrap. For details see igraph
	                documentation. ''')
    opts.add_option('-C','--mapmodules',dest='mapmodules',action="store",default=False,
                    help='''Map the Modules in the PDB provided. The PDB and chain in which to map, have to be provided
                    comma separated. IT ONLY WORKS ON PROTEIN STRUCTURES''')
    opts.add_option('-H','--hierarchy',dest='hierarchy',action="store_true",
                    default=False, help='''Compute hierarchical modules''') 	
    opts.add_option('-R','--rand',dest='rand',action="store_false",
                       default=True, help='''Do not test for randomness of the correlations''')     
    options, args = opts.parse_args()

    prefix = args[0]
    # Introduce the program and write parameters to a file and screen ################
    mc = _Master_of_ceremony(prefix=prefix,**vars(options))
    options._update_loose({'mc':mc})#include mc in the options
    #execute
    main(prefix,**vars(options))