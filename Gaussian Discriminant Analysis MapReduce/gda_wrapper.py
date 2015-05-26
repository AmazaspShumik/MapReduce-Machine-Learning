# -*- coding: utf-8 -*-

import GDA as gda
import numpy as np
from boto.s3.connection import S3Connection
import json
import os

# use if you did not set up this parameters in configuration file
EMR_DEFAULT_PARAMS = ["--ec2-core-instance-bid-price", "0.4", 
                      "--ec2-core-instance-type" ,"m1.small",
                      "--num-ec2-core-instances", "1", 
                      "--ec2-task-instance-bid-price", "0.4", 
                      "--ec2-task-instance-type", "m1.small", 
                      "--num-ec2-task-instances","1"]

# access and secret key
ACCESS_KEY = "YOUR_ACCESS_KEY"
SECRET_KEY = "YOUR_SECRET_KEY"



class GaussianDiscriminantAnalysis(object):
    '''
    Performs Gaussian Discriminant Analysis for classification. Two approaches 
    are available QDA (each class has its own covariance matrix) or LDA (
    covariance matrix is shared).
    
    '''
    
    def __init__(self,targets,dimensions, input_path, output_path, 
                 emr_local = "local", emr_defaults = True):
        self.targets = targets
        self.dimensions = dimensions
        self.input_path = input_path
        self.output_path = output_path
        self.emr_local = emr_local
        self.emr_defaults = emr_defaults
        self.params = {}
        
    def configure(self):
        '''
        Sets configuration parameters to run map reduce job for finding
        parameters of Discriminant Analysis
        '''
        configs = ["--feature-dimensions",str(self.dim),
                   "--targets", json.loads(self.targets),
                   "-r", self.emr_local,
                   "--output-dir",self.output_path,
                   "--no-output",self.input_path]
        configs_new = []
        if self.emr_defaults is True:
            configs_new.extend(EMR_DEFAULT_PARAMS)
        configs_new.extend(configs)
        # start job
        mrJobGDA = gda.GaussianDiscriminantAnalysisMR(configs_new)
        with mrJobGDA.make_runner() as runner:
            runner.run()

    def load_params(self):
        if self.emr_local == "local":
            self.params =  self.local_load_params(self.output_path)
        else:
            self.params =  self.s3_load_params(self.output_path)

                                     
    def s3_load_params(self,s3_path):
        ''' load parameters if they are on amazon s3'''
        path = s3_path.strip("s3://").split("/")
        mybucket = self.conn.get_bucket(path[0]) # connect to s3 bucket
        s3_file_keys = [f for f in mybucket.list(prefix = "/".join(path[1:]))]
        for s3key in s3_file_keys:
            if mybucket.lookup(s3key).size > 0:
                data = s3key.get_contents_as_string()
                params = json.loads(data)
                return params
                
    def local_load_params(self,local_path):
        ''' load paramters if they are on local machine'''
        current_dir = os.getcwd()
        os.chdir(local_path)
        for filename in os.listdir(os.getcwd()):
            if "part-" in filename:
                if os.path.getsize(filename) > 0:
                    with open(filename,"r") as in_file:
                        data = json.load(in_file)
                        os.chdir(current_dir)
                        return data
                        
    def posterior_probs(self, method = ):
        ''' get class probability
        

           method - (str) can have two values either 'QDA' or 'LDA'         
        '''
        
        
        
                 
        