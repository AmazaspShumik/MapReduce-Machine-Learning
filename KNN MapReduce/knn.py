# -*- coding: utf-8 -*-


from mrjob.job import MRJob
from mrjob.protocol import RawValuepProtocol,JSONProtocol,JSONValueProtocol
from mrjob.step import MRStep
import heapq
import csv


################ Helper functions & classes  ##################################

def extract_features(line):
    ''' Extracts data from line of input '''
    data = line.strip.split(",")
    return data[1], [ float(e) for e in data[2:] ]

def dist_euclid(x,y):
    ''' define euclidean distance between two cetor-lists'''
    assert len(x)==len(y), 'vectors should have the same dimensionality'
    return sum([(x[i] - e)**2 for i,e in enumerate(y)])

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
        if self.options.knn_type is None:
            self.option_parser.error("Either 'regression' or 'classification' ")
        else:
            self.knn_type = self.options.knn_type
        # dimensionality
        if self.options.n_neighbours is None:
            self.option_parser.error("You need to specify number of nearest neighbours")
        else:
            self.n_neighbours = self.options.n_neighbours
            
            
    def mapper_knn_init(self):
        ''' 
        Load data point for which classification or regression estimates 
        are required
        '''
        with open("input_points.txt","r") as input_file:
            data = list(csv.reader(input_file))
        self.points = {}
        for data_point in data:
            self.points[tuple([float(e) for e in data_point])] = []
        
            
    def mapper_knn(self,_,line):
        '''
        Calculates nearest neighbours for each point in input set that 
        needs estimation
        '''
        y, features = extract_features(line)
        if len(features) != self.dim:
            raise DimensionalityMismatchError(self.dim,len(features))
        for point in self.points.keys():
            d_inv = -1*dist(features,point)
            observation = tuple([d_inv,features,y])
            # if number of nearest neighbours is smaller than threshold add them
            if len(self.points[point]) < self.n_neighbours:
                self.points[point].append(observation)
                if len(self.points[point]) == self.n_neighbours:
                    heapq.heapify(self.points[point])
            # compare with largest distance and push if it is smaller
            elif d_inv > self.points[point][0]:
                heapq.heapreplace(self.points[point],observaion)
                
                
    def mapper_knn_final(self):
        yield None,self.points
        
        
    def reducer_knn(self,key,points):
        nearest_neigbours = {}
        for data_point in points:
            if len(nearest_neighbours)==0:
                nearest_neighbours = data_point
            else:
                for point in nearest_neighbours:
                    
                    
                    
        
                
                
        
        
        
        
            
    
            
            