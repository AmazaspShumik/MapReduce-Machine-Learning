# -*- coding: utf-8 -*-


from mrjob.job import MRJob
from mrjob.protocol import RawValuepProtocol,JSONProtocol,JSONValueProtocol
from mrjob.step import MRStep
import heapq
import csv
import numpy as np


class NearestNeighbours(object):
    
    
    def __init__(self,points,n_neighbours):
        self.points = {}
        self.n_neighbours = n_neighbours
        for dp in points:
            self.points[dp] = []
            
            
    def dist_euclid(x,y):
       ''' define euclidean distance between two vector-lists'''
       return sum([(x[i] - e)**2 for i,e in enumerate(y)])
        
        
    def process_data_point(self,y,features):
        '''
        
        '''
        for dp in self.points:
           d_inv = -1*self.dist_euclid(features,dp)
           observation = tuple([d_inv,features,y])
           # if number of nearest neighbours is smaller than threshold add them
           if len(self.points[dp]) < self.n_neighbours:
              self.points[dp].append(observation)
              if len(self.points[dp]) == self.n_neighbours:
                 heapq.heapify(self.points[dp])
           # compare with largest distance and push if it is smaller
           else:
              largest_neg_dist = self.points[dp][0][0]
              if d_inv > largest_neg_dist:
                 heapq.heapreplace(self.points[dp],observation)
                 
    def merge_nns(self,mapper_nn_lists):
        for point in self.points.keys():
            # get all priority queues that correspond to data point
            pqs = [e.points[point] for e in mapper_nn_lists if e.points[point]]
            for pq in pqs
                while pq:
                    if len(self.points[point]) < self.n_neighbours:
                        heapq.heappush(self.points[point],heapq.heappop(pq)
                    else:
                        largest_neg_dist = self.points[point][0][0]
                        if pq[0][0] > largest_neg_dist:
                            heapq.heapreplace(self.points[point], heapq.heappop(pq))
                
    def estimate(self,method):
        if mehtod=="regression":
            
        elif method=="classification":
            pass
            
            
            
        
        


class DimensionalityMismatchError(Exception):
    ''' Error for case when dimensionalities do not match'''
    def __init__(self,expected,real):
        self.expected = expected
        self.real = real
        
    def __str__(self):
        error = "Expected  dimensions: "+str(self.expected)+ " observed: "+str(self.real)

###################  MapReduce Job  ########################################### 



class KnnMapReduce(MRJob):
    '''
    K nearest neighbours algorithm for classification and regression.
    
    '''
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTCOL = JSONValueProtocol
    
    
    def __init__(self,*args,**kwargs):
        super(KnnMapReduce,self).__init__(*args,**kwargs)
        
        
        
    #################### load & configure options #############################
    
    def configure_options(self):
        super(KnnMapReduce,self).configure_options()
        self.add_passthrough_option("--dimensionality",
                                    type = int,
                                    help = "dimenisonality of features")
        self.add_passthrough_option("--knn-type",
                                    type = str,
                                    help = "either regression or classification")
        self.add_passthrough_option("--n-neighbours",
                                    type = int,
                                    help = "number of neighbours used in classification or regression")
                                    
                                     
    def load_options(self,args):
        super(KnnMapReduce,self).load_options(args)
        # feature dimensionality
        if self.options.dimensioanlity is None:
            self.option_parser.error("You need to specify feature dimensionality")
        else:
            self.dim = self.options.dimensionality
        # type of knn (either regression or classification)
        if self.options.knn_type != "regression" and self.options.knn_type != "classification":
            self.option_parser.error("Either 'regression' or 'classification' ")
        else:
            self.knn_type = self.options.knn_type
        # dimensionality
        if self.options.n_neighbours is None:
            self.option_parser.error("You need to specify number of nearest neighbours")
        else:
            self.n_neighbours = self.options.n_neighbours
    
    ################# Helper functions for extracting features ################
            
    def extract_features(line):
        ''' Extracts data from line of input '''
        data = line.strip.split(",")
        return (data[1], [ float(e) for e in data[2:] ])
        
    ################# Map - Reduce Job ######################################## 
    
            
    def mapper_knn_init(self):
        ''' 
        Load data point for which classification or regression estimates 
        are required
        '''
        with open("input_points.txt","r") as input_file:
            data = list(csv.reader(input_file))
        self.nns = NearestNeighbours(data)
        
            
    def mapper_knn(self,_,line):
        '''
        Calculates nearest neighbours for each point in input set that 
        needs estimation
        '''
        y, features = extract_features(line)
        if len(features) != self.dim:
            raise DimensionalityMismatchError(self.dim,len(features))
        # for each point select n neighbours that are closest to it
        self.nns.process_data_point(y,features)


    def mapper_knn_final(self):
        yield None,self.nns
        
        
    def reducer_knn(self,key,points):
        nns = None
        for closest_points in points:
            if nns is None:
                nns = closest_points
            else:
                nns = merge_nearest_neugbours(nns,closest_points)
        for point in nns.points:
            # regression
            if self.options.knn_type == "regression":
                estimates = [ observation[-1] for observation in nns.points[point]]
                estimate = sum(estimates)/self.options.n_neighbours
            # classification
            else:
                estimates = {}
                for neg_dist,features,y in nns.points[point]:
                    estimates[y] = estimates.get(y,0) + 1
                estimate,counts = max(estimates.items,key = lambda x: x[-1])
            # format output
            output = list(point)
            output.append(estimate)
            yield None, ",".join([str(e) for e in output])
            
            
    def steps(self):
        return [MRStep(mapper_init  = self.mapper_knn_init,
                       mapper       = self.mapper_knn,
                       mapper_final = self.mapper_knn_final,
                       reducer      = self.reducer_knn)]
                       
if __name__=="__main__"
        
            
                    
            
                    
                    
                    
        
                
                
        
        
        
        
            
    
            
            