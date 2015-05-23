# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:12:00 2015

@author: amazaspshaumyan
"""

from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, RawValueProtocol
from mrjob.step import MRStep
import numpy as np

class LinearRegressionTS(MRJob):
    '''
    Runs standard linear regression for tall and skinny input data.
    Calculates (X_t*X) and X_t*y in sing map-reduce step and then obtains
    coefficients using Moore-Penrose pseudoinverse.
    
    Important!!! Since final computations are performed on single reducer, 
    assumption is that dimensionality of data is relatively small 
    (i.e. input data are tall and skinny).
    
    
    Input: 
          
          Extract relevant features from input line by changing extract_variables
          method. You can add features for non-linear models (like x^2 or exp(x)).
          Current code assumes following input line format:
          
          input line = <dependent variable>, <feature_1>,...,<feature_n>
        
    Output:
     
         output line (without bias term i.e. --bias=false) = <beta_1>,..,<beta_n>
         output line (with bias term) = <beta_1>,<beta_2>,..,<beta_n>,<beta_bias>
    '''
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTOCOL = RawValueProtocol
    
    @staticmethod
    def extract_variables(line, bias = True):
        ''' (str)--(float,[float,float,float...])
        reads line of input and outputs dependent variable and list of 
        explanatory variables.  You may add higher order polynomials and other
        ( You may need to override this method if you use this code for 
        different dataset)
        '''
        data = line.strip().split(",")
        features = [float(e) for e in data[1:]]
        if bias:
            features.append(1.0) # adds bias term
        return (float(data[0]),features) 
    
    
    def __init__(self,*args, **kwargs):
        super(LinearRegressionTS, self).__init__(*args, **kwargs)
        n = self.options.dimension
        self.construct_features = lambda x: LinearRegressionTS.extract_variables(x,self.options.bias)
        self.x_t_x = np.zeros([n,n])
        self.x_t_y = np.zeros(n)
        self.counts = 0
        
    def configure_options(self):
        super(LinearRegressionTS,self).configure_options()
        self.add_passthroogh_option("--dimension", 
                                    type = int,
                                    help = "Number of explanatory variables")
        self.add_passthrough_option("--bias", 
                                    type = bool,
                                    help = "Bias term, bias not included if false ",
                                    default = True)
                                    
    def load_options(self):
        super(LinearRegressionTS,self).load_options()
        if self.options.dimension is None:
            self.option_parser.error("You should define number of explanatory variables")
            
    def mapper_lr(self,_,line):
        y,features = self.construct_features(line)
        if len(features) != self.options.dimension:
            raise 
        self.x_t_x+=np.outer(np.array(features), np.array(features))
        self.x_t_y+=np.array([f*y for f in features])
        self.counts+=1
        
    def mapper_lr_final(self):
        yield 1,("x_t_x", [list(row) for row in self.x_t_x])
        yield 1,("x_t_y", [xy for xy in self.x_t_y])
        yield 1,("counts", self.counts)
        
    def reducer_lr(self,key,values):
        n = self.options.dimension
        observations = 0
        x_t_x = np.zeros([n,n]); x_t_y = np.zeros(n) 
        for val in values:
            if val[0]=="x_t_x":
                x_t_x+=np.array(val[1])
            elif val[0]=="x_t_y":
                x_t_y+=np.array(val[1])
            elif val[0]=="counts":
                observations+=val[1]
        betas =  np.dot(np.linalg.pinv(x_t_x),x_t_y)
        yield None, ",".join([ str(e) for e in betas])
            
    def steps(self):
        return [MRStep(mapper = self.mapper_lr,
                       mapper_final = self.mapper_lr_final,
                       reducer = self.reducer_lr)]
                       
if __name__=="__main__":
    LinearRegressionTS.run()
        

        