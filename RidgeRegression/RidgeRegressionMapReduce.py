# -*- coding: utf-8 -*-


from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol,JSONProtocol,JSONValueProtocol
from mrjob.step import MRStep
import heapq
import csv
import numpy as np
import random



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
    
          -- dimension  - (int)  number of explanatory variables
          -- bias       - (bool) if True regression wil include bias term
    
    
    
    
    
    
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
        self.n = 0
        self.lambdas_cv = self.read_lambdas(self.options.cv_lambdas)
        self.sample_cv = []
        
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
        
    @staticmethod
    def join_mapper_intermediate_stats( mapper_dict_one, mapper_dict_two):
        
        
        
    #----------------------------------------------- Map - Reduce Job -------------------------------------------#
        
    def mapper_ridge(self,_,line):
        y, features = self.extract_features(line)
        x = np.array(features)
        # update instance variables
        if self.options.scaling=="max-min":
            self.max = [max(current_max,features[i]) for i,current_max in enumerate(features)]
            self.min = [max(current_max,features[i]) for i,current_max in enumerate(features)]
        self.mu    = [ av+features[i] for i,av in enumerate(self.mu) ]
        self.x_t_x = np.outer(x,x)
        self.y_av +=y
        self.n    +=1
        # make sample for hold out cross validation set
        rand_priority = random.randrange(start = 0, stop = 100000000)
        observation = (rand_priority,features,y)
        if len(self.sample_cv) < self.cv_size:
            self.sample_cv.append(observation)
            if len(self.sample_cv) == self.cv_size:
                heapq.heapify(self.sample_cv)
        else:
            if observation[0] > self.sample_cv[0][0]:
                heapq.heapreplace(self.sample_cv,observation)
                
    def mapper_ridge_final(self):
        make_json_encodable = lambda x: [list(row) for row in x]
        x_t_x = self.make_json_encodable(self.x_t_x)
        yield None, {"mu":    self.mu,
                     "x_t_x": x_t_x,
                     "y_av":  self.y_av,
                     "n":     self.n}
                     
    def reducer_ridge(self, key, vals):
        # first calculates unscaled solution
        final_summary_stats = {"mu":      [0]*self.dim,
                      "x_t_x":   np.zeros([self.dim,self.dim]),
                      "y_av":    0,
                      "n":       0  }
        for mapper_summary in vals:
            final_summary_stats = self.join
        
        
        
            
        
        
        
        
        
        
        
        
    def steps(self):
        return [MRStep(mapper       = self.mapper_ridge,
                       mapper_final = self.mapper_ridge_final)]
                       
if __name__=="__main__":
    
        
        
        
        
        
