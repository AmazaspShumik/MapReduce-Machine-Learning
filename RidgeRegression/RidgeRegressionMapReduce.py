# -*- coding: utf-8 -*-



from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol,JSONProtocol,JSONValueProtocol
from mrjob.step import MRStep
import heapq
import csv
import numpy as np


class RidgeRegression(MRJob):
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTCOL = JSONValueProtocol
    
    def __init__(self,*args,**kwargs):
        super(RidgeRegression,self).__init__(*args,**kwargs)
        self.mu = np.zeros(self.dim, dtype = np.float)
        self.y_av = 0.0
        self.x_t_x = np.zeros([self.dim,self.dim], dtype = np.float)
        self.n = 0
        
    #----------------- load & configure options ------------------------------#
        
    def configure_options(self):
        super(RidgeRegression,self).configure_options()
        self.add_passthrough_option("--dimension",
                                    type = int,
                                    help = "Number of explanatory variables")
        self.add_passthrough_option("--hold-out-sample-size",
                                    type = int,
                                    help = "Size of sample for hold out cross validation",
                                    default = 1000)
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
            self.option_parser.error("You need to specify ")
        # sample size for hold out cross validation
        self.cv_size = self.options.hold_out_sample_size
        
        
        
