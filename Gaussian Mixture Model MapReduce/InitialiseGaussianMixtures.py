'''
Initialisation step for MapReduce implementation of GMM.

Using MapReduce paradigm samples data from large dataset, so that sample fits
into one machine, then run K-means algorithm on sampled datato find centroids 
and cluster allocation of points.
Cluster allocation of data points is used to get initial parameters for GMM 
(i.e. : mixing coefficients (pdf of latent variable), mean vectors and covariance
matrix for each cluster)
'''

from mrjob.protocol import RawValueProtocol,JSONProtocol, JSONValueProtocol
from mrjob.job import MRJob
from mrjob.step import MRStep
import random
import heapq
import numpy as np



def extract_features(line):
    ''' extracts features from line of input'''
    data = line.strip().split(",")
    return [ float(e) for e in data[1:] ]


#########################  K-means ###########################################


class KmeansInitGMM(object):
    '''
    K-means algorihm for clustering.

    Parameters:
    -----------
    
    clusters           - (int)   number of expected clusters
    dim                - (int)   dimensionality of input
    epsilon            - (float) convergence threshold for k-means
    iteration_limit    - (int)   maximum number of iteration, where each 
                                 iteration consists of e_step and m_step
    data               - (list)  list of lists, where each inner list is 
                                 single data point
    
    '''
    
    def __init__(self, clusters, dim, epsilon, iteration_limit, data):
        self.k = clusters
        self.data = [extract_features(line) for line in data]
        self.m = dim
        self.r = [0]*len(data) # vector of cluster assignments
        self.convergence_epsilon = epsilon
        self.iteration_limit = iteration_limit
        
        
    def loss(self):
        ''' 
        Calculates loss function of K-means
        J =  sum_n[ sum_k [r_n_k*||x_n-mu_k||^2]]]
        '''
        r = self.r
        mu = self.clusters
        J = sum([np.dot((np.array(x)-mu[r[i]]).T,np.array(x)-mu[r[i]]) for i,x in enumerate(self.data)])
        return J
    
    def initialise(self):
        ''' randomly choses points from list'''
        self.clusters = random.sample(self.data,self.k)     
        
    def e_step(self):
        ''' E-step in K means algorithm, finds assignment of points to centroids'''
        for n,data_point in enumerate(self.data):
            min_cl = 0
            min_sq_dist = -1
            for i,cluster in enumerate(self.clusters):
                dist_sq = sum([ (data_point[i]-cluster[i])**2 for i in range(self.m)])
                if min_sq_dist==-1:
                    min_sq_dist = dist_sq
                else:
                    if dist_sq < min_sq_dist:
                        min_sq_dist = dist_sq
                        min_cl = i
            self.r[n] = min_cl

            
    def m_step(self):
        ''' M-step in K-means algorithm, finds centroids that minimise loss function'''
        self.clusters = [[0]*self.m for i in range(self.k)] # update clusters
        cluster_counts = [0]*self.k
        for i,x in enumerate(self.data):
            cluster_counts[self.r[i]]+=1
            self.clusters[self.r[i]] = [self.clusters[self.r[i]][j]+x[j] for j in range(self.m)]
        mean_vector = lambda x,n: [float(el)/n for el in x]
        self.clusters = [mean_vector(self.clusters[i],cluster_counts[i]) for i in range(self.k)] 
            
    
    def run_k_means(self):
        ''' 
        Runs single pass of k-means algorithm
        '''
        self.initialise() # initialise clusters
        next_loss = self.loss() # calculate loss function for initial clusters
        prev_loss = next_loss +2*self.convergence_epsilon
        iteration = 0
        losses = []
        while prev_loss - next_loss > self.convergence_epsilon and iteration < self.iteration_limit:
            self.e_step()
            self.m_step()
            prev_loss = next_loss
            losses.append(prev_loss)
            next_loss = self.loss()
            iteration+=1
        
            
    def run(self, reruns = 10):
        ''' 
        Runs k-means several times and choosed and chooses parameters (mean vectors,
        point cluster allocation) from the k-means run with smallest value of 
        loss function.
        
        (Since loss function is not convex,it is not guaranteed that parameters 
        obtained from single k-means algorithm pass will give global minimum
        of k-means loss function)
        '''
        clusters = [[0]*self.m for i in range(self.k)]
        loss_before = -1
        r = self.r
        for i in range(reruns):
            self.run_k_means()
            loss_new = self.loss()
            if loss_before==-1:
                loss_before = loss_new
                clusters = [el[:] for el in self.clusters]
                r = self.r[:]
            else:
                if loss_new < loss_before:
                    loss_before = loss_new
                    clusters = [el[:] for el in self.clusters]
                    r = self.r[:]
                    
        self.final_r = r
        self.final_clusters = clusters
        
        
    def gmm_params(self):
        ''' 
        Calculates initial parameters for GMM based on cluster allocation of
        points in best K-means
        '''
        total=0
        mixing = [0]*self.k
        covars = [np.zeros([self.m,self.m], dtype = np.float64) for i in range(self.k)]
        mu = [np.zeros(self.m, dtype = np.float64) for i in range(self.k)]
        for i,dp in enumerate(self.data):
            k = self.final_r[i] # cluster
            x = np.array(dp, dtype = np.float64)
            mixing[k]+=1
            total+=1
            mu[k]+=x
            covars[k]+=np.outer(x,x)
        mu = [mu[j]/p for j,p in enumerate(mixing)]
        covars = [1.0/mixing[j]*(covars[j] - mixing[j]*np.outer(mu[j],mu[j])) for j in range(self.k)]
        mixing = [float(p)/total for p in mixing]
        
        matrix_to_list = lambda x: [list(e) for e in x]
        mixing = mixing
        mu = matrix_to_list(mu)
        covariance = [matrix_to_list(e) for e in covars]
        return {"mixing":mixing,"mu":mu,"covariance":covariance}

        
########  intialise parameters of Gaussian Mixture Model #####################


class InitialiseGaussianMixtureMR(MRJob):
    '''
    MapReduce class that initialises parameters of GMM.
    Each mapper assigns random priority to each line of input, chooses n (n = sample size)
    lines with lowest priority level and outputs it.
    Single reducer collects m (where m is number of mappers) lists of size n
    and choses n lines with smallest priority, these final n lines of input
    represent random sample of size n from data. Then k-means algorithm is used
    on sampled data to find parameters for initialising.
           
    Command Line Options:
    ---------------------
    
    --sample-size          - sample size
    --clusters             - number of clusters
    --dimensions           - dimensionality of data
    --kmeans-convergence   - convergence threshold for k-means convergence
    --iteration-limit      - limit on number of iterations for k-means
    --kmeans-reruns        - number of times to run k-means
    
    '''
    
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTOCOL = JSONValueProtocol
    
    def __init__(self,*args,**kwargs):
        super(InitialiseGaussianMixtureMR,self).__init__(*args, **kwargs)
        self.pq = []      
        
    def configure_options(self):
        super(InitialiseGaussianMixtureMR,self).configure_options()
        self.add_passthrough_option("--sample-size",
                                    type= int,
                                    help = "number of elements in sample")
        self.add_passthrough_option("--clusters",
                                    type = int,
                                    help = "number of clusters")
        self.add_passthrough_option("--dimensions",
                                    type = int,
                                    help = "dimensionality of input data")
        self.add_passthrough_option("--kmeans-convergence",
                                    type = float,
                                    default = 0.01,
                                    help = "convergence parameter for K-means loss function")
        self.add_passthrough_option("--iteration-limit",
                                    type = int,
                                    default = 100,
                                    help = "largest number of iterations that k-means algorithm is allowed")
        self.add_passthrough_option("--kmeans-reruns",
                                    type = int,
                                    default = 10,
                                    help = "number of k-means reruns ")
                                    
                                
                                    
    def load_options(self, args):
        super(InitialiseGaussianMixtureMR,self).load_options(args)
        # size of sample for k-means, that will initialise parameters of GMM
        if self.options.sample_size is None:
            self.option_parser.error("You need to specify sample size")
        else:
            self.n = self.options.sample_size
        # number of cluters
        if self.options.clusters is None:
            self.option_parser.error("You need to specify number of clusters")
        else:
            self.k = self.options.clusters
        # dimensionality
        if self.options.dimensions is None:
            self.option_parser.error("You need to specify dimensionality of data")
        else:
            self.dim = self.options.dimensions
            
            
    def mapper_initialise_gmm(self,_,line):
        '''
        Randomly samples n lines of input (where n is sample_size option), by
        assigning random priority level and then choosing n lines of input 
        with smallest priority level
        '''
        r = random.randrange(1000000)
        if len(self.pq) < self.n:
            heapq.heappush(self.pq,(r,line))
        else:
            if self.pq[0][0] < r:
               heapq.heappushpop(self.pq,(r,line))
            
    def mapper_initialise_gmm_final(self):
        yield 1, self.pq
        
    def reducer_kmeans_initialise_gmm(self,key,samples):
        '''
        Subsamples from mapper output and runs K-means algorithm on subsampled
        data to initialise parameters of GMM.        
        '''
        pq_final = []
        for sample in samples:
            for element in sample:
                if len(pq_final) < self.n:
                   pq_final.append(element)
                   if len(pq_final)==self.n:
                       heapq.heapify(pq_final)
                else:
                    if pq_final[0][0] < element[0]:
                        heapq.heappushpop(pq_final,element)
        lines = [line for r,line in pq_final]
        kmeans = KmeansInitGMM(self.k, self.dim, self.options.kmeans_convergence,self.options.iteration_limit,lines)
        kmeans.run(reruns = self.options.kmeans_reruns)
        params = kmeans.gmm_params()
        yield None, params
        
        
    def steps(self):
        return [MRStep(mapper = self.mapper_initialise_gmm,
                       mapper_final = self.mapper_initialise_gmm_final,
                       reducer = self.reducer_kmeans_initialise_gmm)]
                       
if __name__=="__main__":
    InitialiseGaussianMixtureMR.run()
    
    