# -*- coding: utf-8 -*-


from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol,JSONProtocol,JSONValueProtocol
from mrjob.step import MRStep
import heapq
import csv


################# Helper functions & classes ##################################

def dist_euclid(self,x,y):
    ''' define euclidean distance between two vector-lists'''
    return sum([(x[i] - e)**2 for i,e in enumerate(y)])


class DimensionalityMismatchError(Exception):
    ''' Error for case when dimensionalities do not match'''
    def __init__(self,expected,real):
        self.expected = expected
        self.real = real
        
    def __str__(self):
        error = "Expected  dimensions: "+str(self.expected)+ " observed: "+str(self.real)
        return error
        
        
###################  MapReduce Job  ########################################### 



class KnnMapReduce(MRJob):
    '''
    K nearest neighbours algorithm for classification and regression.
    
    
    
    
    '''
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTCOL = JSONValueProtocol
    
        
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
            
    def extract_features(self,line):
        ''' Extracts data from line of input '''
        data = line.strip.split(",")
        return (data[1], [ float(e) for e in data[2:] ])
        
        
    ################# Map - Reduce Job ######################################## 
    
            
    def mapper_knn_init(self):
        ''' 
        Loads data points for which classification or regression estimates 
        are required
        '''
        with open("input_points.txt","r") as input_file:
            data = list(csv.reader(input_file))
        self.points = {}
        for dp in data:
            self.points[dp] = []
        
            
    def mapper_knn(self,_,line):
        '''
        Finds nearest neighbours for each point in set of points that 
        needs to be estimated.
        '''
        y, features = self.extract_features(line)
        if len(features) != self.dim:
            raise DimensionalityMismatchError(self.dim,len(features))
        # for each point select n neighbours that are closest to it
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


    def mapper_knn_final(self):
        yield None,self.points
        
        
    def reducer_knn(self,key,points):
        '''
        Aggregates mapper output
        '''
        merged = None
        for mapper_closest_points in points:
            if merged is None:
                merged = mapper_closest_points
            else:
                for point in merged.keys():
                    pq = mapper_closest_points[point]
                    while pq:
                          if len(merged[point]) < self.n_neighbours:
                             heapq.heappush(merged[point],heapq.heappop(pq))
                          else:
                             largest_neg_dist = merged[point][0][0]
                             if pq[0][0] > largest_neg_dist:
                                heapq.heapreplace(merged[point], heapq.heappop(pq))
        for point in merged.keys():
            # regression
            if self.options.knn_type == "regression":
                estimates = [ observation[-1] for observation in merged[point]]
                estimate = sum(estimates)/self.options.n_neighbours
            # classification
            else:
                estimates = {}
                for neg_dist,features,y in merged[point]:
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
                       
if __name__=="__main__":
    KnnMapReduce.run()
        
            
    
            
            