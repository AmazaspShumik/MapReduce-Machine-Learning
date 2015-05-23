# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:11:14 2015

@author: amazaspshaumyan
"""

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol, JSONProtocol
import numpy as np


class MultivariateDescriptiveStatisticsMR(MRJob):
    ''' 
    Calculates descriptive statistics for multivariate dataset.
    
    Following statistics are calculated:
    
       - Covariance Matrix 
       - Skewness of each variable (measure of assymetry)
       - Kurtosis of each variable (measure of peakedess)
       - Minimum for each variable
       - Maximum for each variable
       - Mean for each variable
          
    Note: accuracy of results were compared on test results with corresponding
    functions in R (min,max,mean,cov,skewness[library(e1071)], kurtosis[library(e1071)])
    '''
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTOCOL = RawValueProtocol
    
    @staticmethod
    def extract_relevant_features(l):
        ''' (str) --> list
        extracts features relevant for task from each line.
        '''
        data = l.strip().split(",")
        return [float(e) for e in data[1:5]]
        
    @staticmethod
    def kurtosis():
        pass
    
    @staticmethod
    def skewness():
        pass
    
    
    def __init__(self, *args, **kwargs):
        super(MultivariateDescriptiveStatisticsMR,self).__init__(*args, **kwargs)
        d = self.dim
        self.n = 0
        self.max,self.min,self.mean = [0]*d,[0]*d,[0]*d
        self.third_order, self.fourth_order = [0]*d, [0]*d
        self.covariates = np.zeros([d,d], dtype = np.float64)
        
        
    def configure_options(self):
        super(MultivariateDescriptiveStatisticsMR,self).configure_options()
        self.add_passthrough_option("--dimensions", type = int, 
                                    help = "Number of columns of data matrix")
                                      
    def load_options(self,args):
        super(MultivariateDescriptiveStatisticsMR,self).load_options(args)
        if self.options.dimensions is None:
            self.option_parser.error("You need specify expected dimensionlity")
        else:
            self.dim =  self.options.dimensions

    def mapper_covar(self,_,line):
        # extract features that you want to analyse
        variables = MultivariateDescriptiveStatisticsMR.extract_relevant_features(line)
        assert(len(variables)==self.dim), "input dimensionality mismatch"
        self.n+=1
        self.max = [max(m, var) for var in variables for m in self.max]
        self.min = [min(m, var) for var in variables for m in self.min]
        self.mean = [s+var for var in variables for s in self.mean]
        self.third_order = [p+var**3 for var in variables for p in self.third_order]
        self.fourth_order = [p+var**4 for var in variables for p in self.fourth_order]
        self.covariates += np.outer(np.array(variables),np.array(variables))
        
    def mapper_covar_final(self):
        yield 1,("max", self.max)
        yield 1,("min", self.min)
        yield 1,("mean", self.mean)
        yield 1,("observations", self.n)
        yield 1,("third order", self.third_order)
        yield 1,("fourth order", self.fourth_order)
        yield 1,("covariates", [list(row) for row in self.covariates])
        
    def reducer_summarise(self,key,values):
        m = self.dim
        p1,max_list,min_list = [0]*m,[0]*m,[0]*m
        p3, p4 = [0]*m,[0]*m
        covar_matr = np.zeros([m,m], dtype = np.float64)
        n = 0
        for val in values:
            if val[0]=="max":
                max_list = [max(max_list[i],var) for i,var in enumerate(val[1])]
            elif val[0]=="min":
                min_list = [min(min_list[i],var) for i,var in enumerate(val[1])]
            elif val[0]=="mean":
                p1 = [p1[i]+var for i,var in enumerate(val[1])]
            elif val[0]=="observations":
                n+=val[1]
            elif val[0]=="third order":
                p3 = [p3[i]+cube for i,cube in enumerate(val[1])]
            elif val[0]=="fourth order":
                p4 = [p4[i]+quad for i,quad in enumerate(val[1])]
            else:
                covar_matr+=np.array(val[1])
        # vector of means
        means = [float(mu)/n for mu in p1]
        # covariance matrix (biased but with lowest MSE)
        covariance = (covar_matr - np.outer(np.array(means),np.array(means))*n)/(n-1)
        # fourth moment: calculate sum((x_i-mean(x))^4) by decomposing it
        p4 = [p4[i]-4*means[i]*p3[i]+6*(means[i]**2)*(covar_matr[i,i])-4*p1[i]*(means[i]**3)+n*means[i]**4 for i in range(m)]
        # third moment: calculate sum((x_i-mean(x))^3) by decompsing it
        p3 = [p3[i]-3*means[i]*covar_matr[i,i]+3*(means[i]**2)*p1[i] - n*means[i]**3 for i in range(m)]     
        # calculate kurtosis for each variable
        kurtosis_standard = [ (kurt/n)/((n-1)*covariance[i,i]/n)**2 -3 for i,kurt in enumerate(p4)]
        # unbised kurtosis (see Joanes and Gill (1998))
        kurtosis_unbiased = [ (kurt*(n+1)+6)*(n-1)/(n-2)/(n-3) for kurt in kurtosis_standard]
        # calculate skewness for each variable
        skewness = [np.sqrt(n*(n-1))/(n-2)*((skew/n)/(((n-1)*covariance[i,i]/n)**1.5)) for i,skew in enumerate(p3)]
        kurtosis_unbiased.append("kurtosis")
        yield None, ",".join([str(kurt) for kurt in kurtosis_unbiased])     
        skewness.append("skewness")
        yield None, ",".join([str(skew) for skew in skewness])
        means.append("mean")
        yield None, ",".join([str(mu) for mu in means])
        min_list.append("min")
        yield None, ",".join([str(min_e) for min_e in min_list])
        max_list.append("max")
        yield None, ",".join([str(max_e) for max_e in max_list])
        for i in range(m):
            row = list(covariance[i,:])
            row.extend(["cov",str(i)])
            yield None, ",".join([str(e) for e in row])
        
        
    def steps(self):
        return [MRStep(mapper = self.mapper_covar,
                       mapper_final = self.mapper_covar_final,
                       reducer = self.reducer_summarise)]
        
if __name__=="__main__":
    MultivariateDescriptiveStatisticsMR.run()