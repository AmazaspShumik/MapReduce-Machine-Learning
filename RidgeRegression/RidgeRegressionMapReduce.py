# -*- coding: utf-8 -*-


from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol,JSONProtocol,JSONValueProtocol
from mrjob.step import MRStep
import heapq
import csv
import numpy as np
import random


# ----------------------------- Helper Classes & Methods --------------------------------

def cholesky_solution_least_squares(part_one, part_two):
    '''Cholesky decomposition '''
    R     = np.linalg.cholesky(part_one)
    z     = np.linalg.solve(R,part_two)
    theta = np.linalg.solve(np.transpose(R),z)
    return theta
    
    
class PrioritySampler(object):
    
    def __init__(self,sample_size):
        self.sample_size = sample_size
        self.sample      = []
        
    def process_observation(self,observation):
        if len(self.sample) < self.sample_size:
            self.sample.append(observation)
            if len(self.sample) == self.sample_size:
                heapq.heapify(self.sample_cv)
        else:
            if observation[0] > self.sample[0][0]:
                heapq.heapreplace(self.sample,observation)
                
    def process_observations(self,observations):
        for observation in observations:
            self.process_observation(observation)
            
            

class RidgeRegressionHoldOutCV(object):
    
    def __init__(self,lambdas, data):
        self.lambdas = lambdas
        self.data    = data
    
    
    def run_ridge_regression(self, lambda_ridge , scaling = None):
        
        def scaler(x, column_scaler):
            m = np.shape(x)[1]
            for i in range(m):
                x[:,i] = column_scaler(x[:,i])
            return x
            
        X,Y  = [],[]
        for observation in self.data:
            features , y = observation[1:]
            X.append(features)
            Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        if scaling == "max-min":
            X = scaler(X,lambda x: x/(np.max(x) - np.min(x)))
        elif scaling == "z-score":
            X = scaler(X,lambda x: (x - np.mean(x))/np.std(x))
        # scale y to account for bias term
        Y = Y - np.mean(Y)
        # in case of max-min and no scaling, we need to substract mean from features
        if scaling != "z-score":
            X = scaler(X, lambda x: x-np.mean(x))
        
    def cv(self, scaling = None):
        err = [ self.run_ridge_regression(lambda_ridge, scaling) for lambda_ridge in self.lambdas]
        lambda_best, err = min([ (self.lambdas[i],err[i]) for i in range(len(self.lambdas)) ], key = lambda t: t[1])
        return lambda_best
            
            
            
class DimensionMismatch(Exception):
    
    def __init__(self,expected,observed):
        self.exp      = expected
        self.obs      = observed
        
    def __str__(self):
        err = "Expected number of observations: "+self.exp+" , observed: "+self.obs
        return err



class RidgeRegression(MRJob):
    '''
    
    Input File:
    -----------
          
          Extract relevant features from input line by changing extract_variables
          method. You can add features for non-linear models (like x^2 or exp(x)).
          Current code assumes following input line format:
          
          input line = <>,<feature_1>,...,<feature_n>,<dependent variable>
          
    Options:
    -----------
    
          --dimension              - (int)  number of explanatory variables
          --scaling                - (str)  'z-score' or 'max-min'
          --hold-out-sample-size   - (int)  size of hold out cross validation set 
          --cv-lambdas             - (str)  name of file containing set of regularisation 
                                            parameters for cross validation
                                            
    '''
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTCOL = JSONValueProtocol
    
    def __init__(self,*args,**kwargs):
        super(RidgeRegression,self).__init__(*args,**kwargs)
        if self.scaling=="max-min":
            self.max = [0]*self.dim
            self.min = [0]*self.dim
        self.mu = [0]*self.dim
        self.y_av = 0.0
        self.x_t_x = np.zeros([self.dim,self.dim], dtype = np.float)
        self.x_t_y = [0]*self.dim
        self.n = 0
        self.lambdas_cv = self.read_lambdas(self.options.cv_lambdas)
        self.sampler = Sampler(self.cv_size)
        
    #------------------------------------------- load & configure options ---------------------------------------#
        
    def configure_options(self):
        super(RidgeRegression,self).configure_options()
        self.add_passthrough_option("--dimension",
                                    type = int,
                                    help = "Number of explanatory variables")
        self.add_passthrough_option("--hold-out-sample-size",
                                    type = int,
                                    help = "Size of sample for hold out cross validation",
                                    default = 1000)
        self.add_passthrough_option("--scaling",
                                    type = str,
                                    help = "Can be 'z-score' or 'max-min' ")
        self.add_file_option("--cv-lambdas",
                             type = "str",
                             help = "Name of file that contains regularisation parameters for cross validation")
                             
    def load_options(self,args):
        super(RidgeRegression,self).load_options(args)
        # dimensionality
        if self.options.dimension is None:
            self.option_parser.error("You need to specify number of explanatory variables")
        else:
            self.dim = self.options.dimension
        # set of lambdas for cross validation
        if self.options.cv_lambdas is None:
            self.option_parser.error("You need to specify name of file with set of regularisation parameters")
        # sample size for hold out cross validation
        self.cv_size = self.options.hold_out_sample_size
        # scaling options
        if self.options.scaling not in [None,'z-score','max-min']:
            self.options_parser.error("You need to define proper scaling ('z-score' or 'max-min')")
            
        
    #----------------------------------------- helper functions ----- --------------------------------------------#
        
    @staticmethod
    def extract_features(line):
        '''
        Extracts dependent variable and features from line of input
        '''
        data = line.strip().split(",")
        features = [float(e) for e in data[1:-1]]
        y = float(data[-1])
        return (y,features)
      
      
    @staticmethod
    def read_lambdas(filename):
        ''' reads regularisation parameters'''
        with open(filename,"r") as csvfile:
            lambdas = list(csv.reader(csvfile))
        return [float(e) for e in lambdas]
        
        
    def join_mapper_intermediate_stats(self, mapper_one, mapper_two):
        '''
        Aggregates mapper outputs
        '''
        mapper_one["mu"]    = [mapper_one["mu"][i] + mapper_two[i] for i in range(self.dim)]
        sum_lists = lambda x,y,n: [x[i] + y[i] for i in range(n)]
        xtx_1, xtx_2 = mapper_one["x_t_x"], mapper_two["x_t_x"] 
        mapper_one["x_t_x"] = [sum_lists(xtx_1[i],xtx_2[i],self.dim) for i in range(self.dim)]
        mapper_one["y_av"] += mapper_two["y_av"]
        mapper_one["n"]    += mapper_two["n"]
        if self.options.scaling == "max-min":
            mapper_one["max"] = [max(mapper_one["max"][i],mapper_two["max"][i]) for i in range(self.dim)]
            mapper_one["min"] = [min(mapper_one["min"][i],mapper_two["min"][i]) for i in range(self.dim)]
        return mapper_one
        
    
    def estimate_params(self,data,lambda_ridge,scaling = None):
        xtx   = np.array(data["x_t_x"])
        xty   = np.array(data["x_t_y"]) 
        mu    = np.array(data["mu"])
        y_av  = data["y_av"]
        n     = data["n"]
        beta_bias   = y_av # (bias terms)
        if scaling is None:
           part_one  = xtx - n*np.outer(mu,mu)+lambda_ridge*np.eye(self.dim)
           part_two  = xty - n*y_av*mu
        elif scaling == "z_score":
           sigma     = 1.0/np.sqrt(np.diag((1.0/n*(xtx-np.outer(mu,mu))))) # vector of standard deviations
           scaler    = np.outer(sigma,sigma)
           part_one  = np.dot(scaler,xtx-n*np.outer(mu,mu)) + lambda_ridge*np.eye(self.dim)
           part_two  = sigma*xty - sigma*mu*y_av*n
        elif scaling == "max-min":
           scale_vec = 1.0/( np.array(data["max"]) - np.array(data["min"]) )
           scaler    = np.outer(scale_vec,scale_vec)
           part_one  = np.dot(scaler,xtx-n*np.outer(mu,mu)) + lambda_ridge*np.eye(self.dim)
           part_two  = scale_vec*xty - scale_vec*mu*y_av*n
        theta = cholesky_solution_least_squares(part_one, part_two)
        return {"bias_term": beta_bias,"theta":list(theta)}
        
        
        
    #----------------------------------------------- Map - Reduce Job -------------------------------------------#
        
    def mapper_ridge(self,_,line):
        y, features = self.extract_features(line)
        x = np.array(features)
        # update instance variables
        if self.options.scaling=="max-min":
            self.max = [max(current_max,features[i]) for i,current_max in enumerate(features)]
            self.min = [max(current_max,features[i]) for i,current_max in enumerate(features)]
        self.mu    = [ av+features[i] for i,av in enumerate(self.mu) ]
        self.x_t_y = [ xty_i + y*features[i] for xty_i,i in enumerate(features)]
        self.x_t_x = np.outer(x,x)
        self.y_av +=y
        self.n    +=1
        # make sample for hold out cross validation set
        rand_priority = random.randrange(start = 0, stop = 100000000)
        observation = (rand_priority,features,y)
        self.sampler.process_observation(observation)
        
                
                
    def mapper_ridge_final(self):
        x_t_x = [list(row) for row in self.x_t_x] # transform numpy array to json-encodable data structure
        intermediate_stats = {"mu":    self.mu,
                              "x_"
                              "x_t_x": x_t_x,
                              "y_av":  self.y_av,
                              "n":     self.n
                             }
        if self.options.scaling == "max-min":
            intermediate_stats["max"] = self.max
            intermediate_stats["min"] = self.min
        yield None, ("stats",intermediate_stats)
        yield None, ("hold_out_cv",self.sampler.sample)
            
                  
                  
    def reducer_ridge(self, key, vals):
        '''
        
        '''
        sampler = Sampler(self.cv_size)
        final_summary_stats = {"mu":      [0]*self.dim,
                               "x_t_x":   [[0]*self.dim for i in range(self.dim)],
                               "x_t_y":   [0]*self.dim,
                               "y_av":    0,
                               "n":       0  }
        for val in vals:
            if val[0]=="stats":
                mapper_summary = val[1]
                final_summary_stats = self.join_mapper_intermediate_stats(final_summary_stats,mapper_summary)
            else:
                sampler.process_observations(val[1])
        # for each scaling type use cross validation to verify best lambda
        # then use it on all data (including cv set) to find parameters
        ridge   = RidgeRegressionHoldOutCV(self.lambdas, sampler.sample)
        best_lambda = ridge.cv(self.options.scaling)
        yield None, self.estimate_params(final_summary_stats,best_lambda,self.options.scaling)

            
            
    def steps(self):
        return [MRStep(mapper       = self.mapper_ridge,
                       mapper_final = self.mapper_ridge_final,
                       reducer      = self.reducer_ridge)]
                       
if __name__=="__main__":
    RidgeRegression.run()
    
        
        
        
        
        
