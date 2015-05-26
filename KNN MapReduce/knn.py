# -*- coding: utf-8 -*-


from mrjob.job import MRJob
from mrjob.protocol import RawValuepProtocol,JSONProtocol,JSONValueProtocol
from mrjob.step import MRStep


################ Helper functions & classes  ##################################









###################  MapReduce Job  ########################################### 



class KnnMapReduce(MRJob):
    '''
    K nearest neighbours algorithm for classification and regression.
    
    '''
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTCOL = JSONValueProtocol
    
    
    def __init__(self,*args,**kwargs):
        self.
        self.
    
    
    
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
        # type of 
        if self.options.clusters is None:
            self.option_parser.error("You need to specify number of clusters")
        else:
            self.k = self.options.clusters
        # dimensionality
        if self.options.dimensions is None:
            self.option_parser.error("You need to specify dimensionality of data")
        else:
            self.dim = self.options.dimensions
            