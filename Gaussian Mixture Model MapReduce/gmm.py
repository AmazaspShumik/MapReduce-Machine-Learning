# -*- coding: utf-8 -*-
"""
Gaussian Mixture Model on EMR


"""

import InitialiseGaussianMixtures as gmm_init
import IterationGaussianMixtureMR as gmm_iterator
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


                      
def dist_tot(mu_before, mu_after):
    ''' calculates sum of distances between list of vectors '''
    diffs = [np.array(mu_before[i])-np.array(mu) for i,mu in enumerate(mu_after)]
    return sum([np.sqrt(np.dot(mu_diff.T,mu_diff)) for mu_diff in diffs])
    
    
    
class Runner(object):
    
    """
    (i.e. sample and run K-means on sample to determine initial parameters
        )
    """
    
    def __init__(self,d,k,init_eps,sample_size,init_iteration_limit,
                 iteration_eps,em_iteration_limit, input_path, 
                 output_path,emr_local = "local", emr_defaults = False):
        self.dim = d                                        # dimensionality of data
        self.clusters = k                                   # number of expected clusters
        self.init_eps = init_eps                            # convergence threshold for K-means on initialisation step
        self.init_iteration_limit = init_iteration_limit    # limit for iterations for K-means on initial step
        self.iteration_eps = iteration_eps                  # convergence threshold for EM parameter
        self.em_iteration_limit = em_iteration_limit        # maximum number of iterations of EM algorithm
        self.input_path = input_path
        self.output_path = output_path
        self.sample_size = sample_size
        self.emr_defaults = emr_defaults
        assert emr_local=='emr' or emr_local=='local', " 'emr_local' should be either 'emr' or 'local' "
        self.emr_local = emr_local
        if self.emr_local == "emr":
            self.conn = S3Connection(aws_access_key_id = ACCESS_KEY,
                                     aws_secret_access_key = SECRET_KEY)
            
        
        
    ############### Initialisation of GMM parameters ##########################
        
        
    def config_and_run_init_step(self):
        ''' 
        Sets configuration paramters to run initial step of GMM algorithm.
        By default job will run in 'local' mode
        '''
        # set configuration
        init_configs = ["--dimensions",str(self.dim),
                        "--sample-size",str(self.sample_size),
                        "--clusters",str(self.clusters),
                        "--iteration-limit",str(self.init_iteration_limit),
                        "--kmeans-convergence",str(self.init_eps),
                        "-r", self.emr_local,
                        "--output-dir","_".join([self.output_path,"0"]),
                        "--no-output",self.input_path]
        init_configs_new = []
        if self.emr_defaults is True:
            init_configs_new.extend(EMR_DEFAULT_PARAMS[:])
        init_configs_new.extend(init_configs)
        # start job 
        mrJobInitStep = gmm_init.InitialiseGaussianMixtureMR(init_configs_new)
        with mrJobInitStep.make_runner() as runner:
            runner.run()
            
    
    #######################  Iterations of EM-algorithm  ######################
    
    @staticmethod
    def delta_stop_iterate(old_params,new_params):
        '''
        
        '''
        mu_old = old_params["mu"]
        mu_new = new_params["mu"]
        delta = dist_tot(mu_new,mu_old)
        return delta
        
        
    
    def iterate_em(self):
        '''
        Performs em iterations until convergence
        '''
        delta = 10
        get_params = lambda p,i: self.load_params("_".join([p,str(i)])) # get parameters from previous iter.
        old_params = get_params(self.output_path,0)
        iteration = 1
        while delta > self.iteration_eps and iteration < self.em_iteration_limit:
            self.config_and_run_iter_step(iteration, json.dumps(old_params))
            new_params = get_params(self.output_path,iteration)
            delta = self.delta_stop_iterate(old_params,new_params)
            iteration+=1
            old_params = new_params
          
        

    def config_and_run_iter_step(self,iteration, parameters):
        '''
        Configure parameters to run single iteration of EM algorithm 
        (each iteration consists of E-step and M-step)
        '''
        iter_configs = [ "--dimensions",str(self.dim),
                         "--clusters",str(self.clusters),
                         "--parameters", parameters,
                         "-r", self.emr_local,
                         "--output-dir","_".join([self.output_path,str(iteration)]),
                         "--no-output",self.input_path ]
        iter_configs_new = []
        if self.emr_defaults is True:
            iter_configs_new.extend(EMR_DEFAULT_PARAMS)
        iter_configs_new.extend(iter_configs)
        # start job
        mrJobIterStep = gmm_iterator.IterationGaussianMixtureMR(iter_configs_new)
        with mrJobIterStep.make_runner() as runner:
            runner.run()
            
                                            
    def load_params(self,path):
        if self.emr_local == "local":
            return self.local_load_params(path)
        return self.s3_load_params(path)

                                     
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
                    
    def folder_cleanup(self):
        pass
    
    
    def main_run():
        pass
        
            
        
if __name__=="__main__":
    d = 2
    k = 2
    init_eps = 0.01
    sample_size = 100
    init_iteration_limit = 20
    iteration_eps = 0.01
    em_iteration_limit = 10
    
    #input_path = "/Users/amazaspshaumyan/Desktop/MapReduceAlgorithms/map_reduce/gmm_test_data.txt"
    #output_path = "/Users/amazaspshaumyan/Desktop/MapReduceAlgorithms/map_reduce/gmm_test_final_iteration"
    output_path = "s3://test-map-reduce-movielabs/expectation_maximization_clients/gmm_test_output_initial_test"
    input_path = "s3://test-map-reduce-movielabs/expectation_maximization_clients/gmm_test_data.txt"
    emr_local = "emr"
    emr_defaults = True
    gmm_mr = Runner(d,k,init_eps,sample_size,init_iteration_limit,
                 iteration_eps,em_iteration_limit, input_path, 
                 output_path,emr_local, emr_defaults)
    gmm_mr.config_and_run_init_step()
    gmm_mr.iterate_em()
 
                
        
    
        
        
    
            
        
        
    
    
    
    
         
        
    
    
    
    

                                                               
    
    
    