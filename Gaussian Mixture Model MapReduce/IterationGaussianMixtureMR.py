
from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, RawValueProtocol, JSONValueProtocol
from mrjob.step import MRStep
import json
import numpy as np


def multivar_gauss_pdf(x, mu, cov):
    '''
    Caculates the multivariate normal density (pdf)
    
    Input:
    
    x - numpy array of a "d x 1" sample vector
    mu - numpy array of a "d x 1" mean vector
    cov - numpy array of a d x d" covariance matrix
    
    (where d - dimensionality of data)

    Output:
            - (float) probability of x given parameters of 
                     Gaussian Distribution
    '''
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * np.dot(np.dot((x-mu).T,(np.linalg.inv(cov))),(x-mu))
    return float(part1 * np.exp(part2))
        

def responsibility(x,mu,cov,p,K):
    ''' 
    Calculates conditional probability of latent variable given
    observed data and parameters
    
    Input:
    
    x - numpy array of a "d x 1" sample vector
    mu - list of length "K" of lists "d x 1" mean vector 
    cov - list of length "K" numpy arrays each "d x d" covariance matrix
    p - list of floats, each float prior probability of cluster
    K - number of clusters (values of latent variables)
    
    (where d - dimensionality of data)
    
    Output:
          - list of floats, each element of list is responsibility corresponding 
            to x and relevant latent variable valiue
    '''
    resps = [p[k]*multivar_gauss_pdf(x,np.array(mu[k]),np.array(cov[k])) for k in range(K)]
    p_x = sum(resps)
    return [float(r_k)/p_x for r_k in resps]
    

def extract_features(line):
    ''' extracts features from line of input'''
    data = line.strip().split(",")
    return [ float(e) for e in data[1:] ]
    
    
def make_json_encodable(mixing, means, covar):
    """ transforms """
    matrix_to_list = lambda x: [list(e) for e in x]
    mixing = mixing
    means = matrix_to_list(means)
    covariance = [matrix_to_list(e) for e in covar]
    return {"mixing":mixing,"mu":means,"covariance":covariance}


   
class IterationGaussianMixtureMR(MRJob):
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTOCOL = JSONValueProtocol
        

    def __init__(self,*args,**kwargs):
        super(IterationGaussianMixtureMR,self).__init__(*args,**kwargs)
        # sum of responsibilities for each cluster & number of observations
        self.resp_sum = [0]*self.clusters
        self.N = 0
        # sum of observations weighted by reponsibility 
        self.resp_w_sum = [np.zeros(self.dim, dtype = np.float64) for i in range(self.clusters)]
        # sum of x_n*x_n_t (outer products) weighted by reponsibility
        self.resp_w_cov = [np.zeros([self.dim,self.dim], dtype = np.float64) for i in range(self.clusters)]   
        
        
    def configure_options(self):
        super(IterationGaussianMixtureMR,self).configure_options()
        self.add_passthrough_option("--dimensions",
                                    type = int,
                                    help = "dimensionality of input data")
        self.add_passthrough_option("--clusters",
                                    type = int,
                                    help = "number of clusters")
        self.add_passthrough_option("--parameters",
                             type = str,
                             help = "file with parameters from previous iteration")
    
    
    def load_options(self,args):
        super(IterationGaussianMixtureMR,self).load_options(args)
        # number of clusters
        if self.options.clusters is None:
            self.option_parser.error("You need to specify number of clusters")
        else:
            self.clusters = self.options.clusters
        # data dimensionality
        if self.options.dimensions is None:
            self.option_parser.error("You need to specify dimensionality of data")
        else:
            self.dim = self.options.dimensions
        # filename where parameters from previous iteration are saved
        if self.options.parameters is None:
            self.option_parser.error("You need to load file with distribution parameters")
            
    def mapper_gmm_init(self):
        params = json.loads(self.options.parameters)
        self.mu = params["mu"]
        self.covar = params["covariance"]
        self.mixing = params["mixing"]
    
    def mapper_gmm(self,_,line):
        features = extract_features(line)
        assert(len(features)==self.dim), "dimension mismatch"
        x = np.array(features)
        r_n = responsibility(x,self.mu,self.covar,self.mixing,self.clusters) # responsibilities
        self.resp_sum = [self.resp_sum[i]+r_n_k for i,r_n_k in enumerate(r_n)]
        self.resp_w_sum = [w_sum + r_n[i]*x for i,w_sum in enumerate(self.resp_w_sum)]
        self.resp_w_cov = [w_covar+r_n[i]*np.outer(x,x) for i,w_covar in enumerate(self.resp_w_cov)]
        self.N+=1
        
    def mapper_final_gmm(self):
        matrix_to_list = lambda x: [list(e) for e in x]
        yield 1,("r_sum", self.resp_sum)                                       # sum of responsibilities
        yield 1,("r_w_sum", [list(e) for e in self.resp_w_sum])                # sum of observations weighted by responsibility
        yield 1,("r_w_cov", [ matrix_to_list(cov) for cov in self.resp_w_cov]) # covariates weighted by responsibility
        yield 1,("total", self.N)                                              # number of observations
        
    
    def reducer_gmm(self,key, values):
        N = 0;
        r_sum = [0]*self.clusters
        r_w_sum = [np.zeros(self.dim, dtype = np.float64) for i in range(self.clusters)]
        r_w_cov = [np.zeros([self.dim,self.dim], dtype = np.float64) for i in range(self.clusters)]
        for value in values:
            if value[0]=="r_sum":
                r_sum = [r_sum[i]+gamma for i,gamma in enumerate(value[1])]
            elif value[0]=="r_w_sum":
                r_w_sum = [r_w_sum[i]+np.array(r_w_new, dtype = np.float64) for i,r_w_new in enumerate(value[1])]
            elif value[0]=="r_w_cov":
                r_w_cov = [ r_w_cov[i] + np.array(cov) for i,cov in enumerate(value[1])]
            elif value[0]=="total":
                N+=value[1]
        mixing = [float(gamma)/N for gamma in r_sum]
        means =  [1.0/r_sum[i]*r_w_sum[i] for i, gamma in enumerate(mixing)]
        covar =  [ 1.0/r_sum[k]*r_w_cov_k - np.outer(means[k],means[k]) for k,r_w_cov_k in enumerate(r_w_cov)]     
        yield None, make_json_encodable(mixing,means,covar)

    def steps(self):
        return [MRStep(mapper_init = self.mapper_gmm_init,
                       mapper = self.mapper_gmm, 
                       mapper_final = self.mapper_final_gmm,
                       reducer = self.reducer_gmm)]
                       
if __name__=="__main__":
    IterationGaussianMixtureMR.run()
    
    