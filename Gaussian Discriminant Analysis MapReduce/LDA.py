# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:01:15 2015

@author: amazaspshaumyan
"""

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol,JSONProtocol
import numpy as np
import json

def extract_features(line):
    data = line.strip.split(",")
    return data[1], data[2:] 
    
def write_file(filename):
    
    


class GaussianDiscriminantAnalysisMR(MRJob):
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTCOL = JSONProtocol
    
    
    def __init__(self,*args,**kwargs):
        super(GaussianDiscriminantAnalysisMR,self).__init__(*args,**kwargs)
        classes = len(self.targets)
        self.priors = [0]*n
        self.means = [np.zeros(self.dim) for i in range(n)]
        self.covariate = np.zeros([self.dim,self.dim])
        self.total = 0
        self.targets_to_index = {}

        
        
    def configure_options(self):
        super(GaussianDiscriminantAnalysisMR,self).configure_options()
        self.add_passthrough_option("--feature-dimensions", 
                                      type = int,
                                      help = "dimensionality of features")
        self.add_passthrough_option("--targets",
                                    type = str,
                                    help = "targets")
        self.add_passthrough_option("--outfile",
                                    type = str,
                                    help = "output file with parameter values")
                                      
    def load_options(self,args):
        super(GaussianDiscriminantAnalysisMR,self).load_options(args)
        if self.options.feature_dimension is None:
            self.option_parser.error("You must specify dimensionality of data")
        else:
            self.dim = self.options.feature_dimension
        if self.options.targets is None:
            self.option_parser.error("You must specify targets")
        else:
            self.targets = self.options.targets
        if self.optiions.outfile is None:
            self.option_parser.error("You must specify file where to write parameters")
    
    def mapper_gda(self,_,line):
        
        # extract features
        data = self.feature_extractor(line)
        y = data[0]; features = [float(e) for e in data[1:]]
        
        # raise errors if parameters are not specified
        if self.options.target_one is None:
            self.option.parser.error("You must specify --target-one")
        if self.options.target_two is None:
            self.options.parse.error("You must specify --target-two")

            
            
        # set dimensionality if feature dimensionality is not given
        n = len(features)
                    
        # if there is dimensionality mismatch raise an error
        if n!=self.options.feature_dimension:
            self.option_parser.error("Dimensionality mismatch, lines of input had different length")
        
        # Update all parameters:
        # 1) Update number of observations
        self.n+=1
        # 2) Update sum vectors and priors
        if y==self.options.target_one:
            self.priors[0]+=1
            self.sums_class_one = [self.sums_first_class[i]+feature_val for (i,feature_val) in enumerate(features)]
        elif y==self.options.target_two:
            self.priors[1]+=1
            self.sums_class_two = [self.sum_second_class[i]+feature_val for (i,feature_val) in enumerate(features)]
        else:
            self.option_parser.error("There is more than two type of target values in input files")
        # 3) update  covariate matrix
        self.covariate_matrix = self.covariate_matrix+np.ones([n,n])*np.array(features)
        
    def mapper_final_lda(self):
        yield 1,("total observations", self.n)
        yield 1,("class counts", self.priors)
        yield 1,("class one sum", self.sums_class_one)
        yield 1,("class two sum", self.sums_class_two)
        yield 1,("covariate matrix", self.covariate_matrix)
        
        
    def reducer_lda_parameters(self,key, parameters):
        # summarise data
        total = 0
        class_counts = [0,0]
        dim = self.options.feature_dimension
        sum_one = [0]*dim; sum_two = [0]*dim; covariate = np.zeros([dim,dim]);
        for param_name, param_val  in parameters:
            if param_name == "total observations":
                total+=param_val
            elif param_name == "class counts":
                class_counts[0]+=param_val[0]
                class_counts[1]+=param_val[1]
            elif param_val == "class one sum":
                    sum_one = [sum_one[i]+val for (i,val) in enumerate(param_val)]
            elif param_val == "class two sum":
                    sum_two = [sum_one[i]+val for (i,val) in enumerate(param_val)]
            else:
                    covariate+=param_val
        # calculate mean for each class
        mu_one, mu_two = [float(e)/total for e in sum_one], [float(e)/total for e in sum_two]
        # calculate prior probabilities for each class (based on frequency of occurence for each class)
        prior_one, prior_two = [float(e)/total for e in class_counts]
        # calculate single 
        
    def steps(self):
        return [MRStep(mapper = self.)]
        
        
        
        
                
        
        
            
        