

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


class KmeansInitGMM(object):
    '''
    K-means algorihm that is used to initialise parameters of GMM.
    This K-means is runned on small subset of data.
    '''
    
    def __init__(self, clusters, dim, epsilon, iteration_limit, data):
        self.k = clusters
        self.data = [extract_features(line) for line in data]
        self.m = dim
        self.r = [0]*len(data) # vector of cluster assignments
        self.convergence_epsilon = epsilon
        self.iteration_limit = iteration_limit
        
        
    def loss(self):
        ''' calculates loss function of K-means
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
        ''' M-step in K-means algorithm, finds centroids that will minimise 
            loss function
        '''
        self.clusters = [[0]*self.m for i in range(self.k)] # update clusters
        cluster_counts = [0]*self.k
        for i,x in enumerate(self.data):
            cluster_counts[self.r[i]]+=1
            self.clusters[self.r[i]] = [self.clusters[self.r[i]][j]+x[j] for j in range(self.m)]
        mean_vector = lambda x,n: [float(el)/n for el in x]
        self.clusters = [mean_vector(self.clusters[i],cluster_counts[i]) for i in range(self.k)] 
            
    
    def run_k_means(self):
        ''' single pass of k-means algorithm. Since loss function is not convex,
            it is not guaranteed that resulted output will minimise k-means
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
        ''' Reruns k-means algorithm to find optimal solution'''
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
        ''' calculates initial parameters for GMM based on best k-means result'''
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

        
######## map-reduce job, intialises parameters of Gaussian Mixture Model ######


class InitialiseGaussianMixtureMR(MRJob):
    
    
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
        r = random.randrange(1000000)
        if len(self.pq) < self.n:
            heapq.heappush(self.pq,(r,line))
        else:
            if self.pq[0][0] < r:
               heapq.heappushpop(self.pq,(r,line))
            
    def mapper_initialise_gmm_final(self):
        yield 1, self.pq
        
    def reducer_kmeans_initialise_gmm(self,key,samples):
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
        kmeans.run(reruns = 10)
        params = kmeans.gmm_params()
        yield None, params
        
        
    def steps(self):
        return [MRStep(mapper = self.mapper_initialise_gmm,
                       mapper_final = self.mapper_initialise_gmm_final,
                       reducer = self.reducer_kmeans_initialise_gmm)]
                       
if __name__=="__main__":
    InitialiseGaussianMixtureMR.run()
    
    
    