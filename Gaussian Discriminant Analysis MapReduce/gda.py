# -*- coding: utf-8 -*-


from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol,JSONProtocol, JSONValueProtocol
import numpy as np
import json

################### Helper function & classes #################################


def extract_features(line):
    ''' Extracts data from line of input '''
    data = line.strip.split(",")
    return data[1], data[2:]
    
def matrix_to_list(input_data):
    return [list(e) for e in input_data]
    
class DimensionalityMismatchError(Exception):
    ''' Error when dimensionalities do not match '''
    def __init__(self,expected,real):
        self.exp = expected
        self.real = real
        
    def __str__(self):
        error = "Expected number of dimensions: "+str(self.exp)+" observed: "+ str(self.real)
        return error
        
        
class TargetValueError(Exception):
    ''' Error for target values '''
    def __init__(self,observed):
        self.observed = observed
    
    def __str__(self):
        error = "Observed value "+str(self.e) + " is not target value"
        return error
        
        
####################### MapReduce Job  ########################################


class GaussianDiscriminantAnalysisMR(MRJob):
    '''
    Calculates parameters required for Linear Discriminant Analysis and 
    Quadratic Discrminant Analysis. 
    
    
    Command Line Options:
    ---------------------
    
    --feature-dimensions  - dimensionality of features (dependent variables)
    --targets             - list of all valid target values (json-encoded list)
    '''
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTCOL = JSONValueProtocol
    
    
    def __init__(self,*args,**kwargs):
        super(GaussianDiscriminantAnalysisMR,self).__init__(*args,**kwargs)
        self.k = len(self.targets)
        self.priors = [0]*self.k
        self.means = [np.zeros(self.dim) for i in range(self.k)]
        self.covariate = [np.zeros([self.dim,self.dim]) for i in range(self.k)]
        self.total = 0
        self.targets = json.loads(self.targest)
        self.target_set = set(self.targets)
        self.target_to_index = {}
        for i,target in enumerate(self.targets):
            self.target_to_index[target] = i
            
        
    def configure_options(self):
        super(GaussianDiscriminantAnalysisMR,self).configure_options()
        self.add_passthrough_option("--feature-dimensions", 
                                      type = int,
                                      help = "dimensionality of features")
        self.add_passthrough_option("--targets",
                                    type = str,
                                    help = "targets")

                                      
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
            
    
    def mapper_gda(self,_,line):
        '''
        Calculates and summarise intermediate values for each mapper.
        (Intermediate values include number of observations in each class,
        total number of observations etc. )
        '''
        y,features = extract_features(line)
        n = len(features)
        x = np.array(features)
        index = self.target_to_index[y]
        # error if dimensionalities do not match
        if len(features) != self.dim:           
            raise DimensionalityMismatchError(self.dim,n)
        # targets are not in set of targets defined
        if y not in self.target_set:
            raise TargetValueError(y)
        self.total+=1
        self.means[index] += x
        self.covariate[index] += np.outer(x,x)
        self.priors[index] += 1
        
        
    def mapper_final_gda(self):
        '''Outputs data summarised for each mapper to reducer'''
        yield 1,{ "total": self.total,
                  "class counts": self.priors,
                  "means": matrix_to_list(self.means),
                  "covariates": [matrix_to_list(e) for e in self.covariate]}
        
        
    def reducer_gda_parameters(self,key, parameters):
        ''' Summarises intermediate values produced by each mapper to get final parameters '''
        all_parameters = {}
        # sum two lists (each list has length = number of classes)
        vec_sum = lambda x,y: [x[i]+y[i] for i in range(self.k)]
        # sum two list of lists
        list_of_vec_sum = lambda x,y: [vec_sum(x[i],y[i]) for i in range(self.k)]
        list_of_matrix_sum = lambda x,y: [list_of_vec_sum(x[i],y[i]) for i in range(self.k)]
        # summarise parameters produced by each mapper
        for parameter in parameters:
            if len(all_parameters)==0:
                all_parameters = parameters
            else:
                all_parameters["total"]+=parameters["total"]
                all_parameters["class counts"] = vec_sum(parameter["class counts"],all_parameters["class counts"])
                all_parameters["means"] = list_of_vec_sum(parameter["means"],all_parameters["means"])
                all_parameters["covariates"] = list_of_matrix_sum(parameter["covariates"],all_parameters["covariates"])
        # calculate final parameters
        for i in range(self.k):
            all_parameters["means"][i] = float(all_parameters["means"][i])/all_parameters["class counts"][i]
            mu = np.array(all_parameters["means"][i])
            all_parameters["covariates"][i] = np.array(all_parameters["covariates"][i]) - all_parameters["class counts"][i]*np.outer(mu,mu)
            all_parameters["covariates"][i] = matrix_to_list(all_parameters["covariates"][i])
        yield None, all_parameters
            
            
    def steps(self):
        return [MRStep(mapper = self.mapper_gda,
                       mapper_final = self.mapper_final_gda,
                       reducer = self.reducer_lda_parameters)]
                       
                       
if __name__=="__main__":
    GaussianDiscriminantAnalysisMR.run()
                       
        
        