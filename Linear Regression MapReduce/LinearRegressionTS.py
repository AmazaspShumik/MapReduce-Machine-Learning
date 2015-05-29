# -*- coding: utf-8 -*-

from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, RawValueProtocol
from mrjob.step import MRStep
import numpy as np


######################## Helper Methods and Classes  ##########################


def cholesky_solution_linear_regression(x_t_x,x_t_y):
    '''
    Finds parameters of regression through Cholesky decomposition,
    given sample covariance of explanatory variables and covariance 
    between explanatory variable and dependent variable.
    
    Paramaters:
    -----------
    x_t_x    - numpy array of size 'm x m', represents sample covariance of explanatory variables
    x_t_y    - numpy array of size 'm x 1', represent covariance between expalanatory and dependent variable
    
    Output:
    -------
    Theta   - list of size m, represents values of coefficients 
    
    '''
    # L*L.T*Theta = x_t_y
    L = np.linalg.cholesky(x_t_x)
    #  solve L*z = x_t_y
    z = np.linalg.solve(L,x_t_y)
    #  solve L.T*Theta = z
    theta = np.linalg.solve(np.transpose(L,z))
    return theta
    
   

class DimensionMismatchError(Exception):

    def __init__(self,expected,observed):
        self.exp = expected
        self.obs = observed
        
    def __str__(self):
        err = "Expected number of dimensions: "+str(self.exp)+", observed: "+str(self.obs)
        return err
    


############################## Map Reduce Job #################################



class LinearRegressionTS(MRJob):
    '''
    Calculates sample covariance matix of explanatory variables (x_t_x) and 
    vector of covariances between dependent variable expanatory variables (x_t_y)
    in single map reduce pass and then uses cholesky decomposition to
    obtain values of regression parameters.
    
    
    Important!!! Since final computations are performed on single reducer, 
    assumption is that dimensionality of data is relatively small i.e. input 
    matrix is tall and skinny.
    
    
    Input File:
    -----------
          
          Extract relevant features from input line by changing extract_variables
          method. You can add features for non-linear models (like x^2 or exp(x)).
          Current code assumes following input line format:
          
          input line = <dependent variable>, <feature_1>,...,<feature_n>
          
    Options:
    -----------
    
          -- dimension  - (int)  number of explanatory variables
          -- bias       - (bool) if True regression wil include bias term
    
    Output:
    -----------
          json-encoded list of parameters
    '''
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTOCOL = RawValueProtocol
       
    
    def __init__(self,*args, **kwargs):
        super(LinearRegressionTS, self).__init__(*args, **kwargs)
        n = self.options.dimension
        self.construct_features = lambda x: LinearRegressionTS.extract_variables(x,self.options.bias)
        self.x_t_x = np.zeros([n,n])
        self.x_t_y = np.zeros(n)
        self.counts = 0
        
    #--------------------------- feature extraction --------------------------#
        
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
        
        
    #---------------------------- Options ------------------------------------#
        
    def configure_options(self):
        ''' Additional options'''
        super(LinearRegressionTS,self).configure_options()
        self.add_passthroogh_option("--dimension", 
                                    type = int,
                                    help = "Number of explanatory variables")
        self.add_passthrough_option("--bias", 
                                    type = bool,
                                    help = "Bias term, bias not included if false ",
                                    default = True)
                                    
    def load_options(self):
        ''' Loads and checks whether options are provided'''
        super(LinearRegressionTS,self).load_options()
        if self.options.dimension is None:
            self.option_parser.error("You should define number of explanatory variables")
        else:
            self.dim = self.options.dimension
            
            
    #------------------------ map-reduce steps -------------------------------#
            
            
    def mapper_lr(self,_,line):
        '''
        Calculates x_t_x and x_t_y for data processed by each mapper
        '''
        y,features = self.construct_features(line)
        x = np.array(features)
        if len(features) != self.dim:
            raise DimensionMismatchError(self.dim,len(features))
        self.x_t_x+=np.outer(x, x)
        self.x_t_y+=y*x
        self.counts+=1
        
    def mapper_lr_final(self):
        '''
        Transforms numpy arrays x_t_x and x_t_y into json-encodable list format
        and sends to reducer
        '''
        yield 1,("x_t_x", [list(row) for row in self.x_t_x])
        yield 1,("x_t_y", [xy for xy in self.x_t_y])
        yield 1,("counts", self.counts)
        
    def reducer_lr(self,key,values):
        '''
        Aggregates results produced by each mapper and obtains x_t_x and x_t_y
        for all data, then using cholesky decomposition obtains parameters of 
        linear regression.
        '''
        n = self.dim
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
        '''Defines map-reduce steps '''
        return [MRStep(mapper = self.mapper_lr,
                       mapper_final = self.mapper_lr_final,
                       reducer = self.reducer_lr)]
                       
if __name__=="__main__":
    LinearRegressionTS.run()
        

        