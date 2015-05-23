# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:01:20 2015

@author: amazaspshaumyan
"""

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol, JSONProtocol
import random
import heapq


class SimpleRandomSampleNoReplacementMR(MRJob):
    ''' Simple Random Sampling without replacement for relatively small sample
    sizes. 
    Do not use for large sample sizes that can not fit in memory (current code
    uses only one reducer)
    
    Each line in input data is assigned random priority then n lines with largest
    corresponding priorities are selected (where n is size of random sample)

    '''
    
    INPUT_PROTOCOL = RawValueProtocol
    
    INTERNAL_PROTOCOL = JSONProtocol
    
    OUTPUT_PROTOCOL = RawValueProtocol
    
    def __init__(self,*args,**kwargs):
        super(SimpleRandomSampleNoReplacementMR,self).__init__(*args, **kwargs)
        self.pq = []      
        
    def configure_options(self):
        super(SimpleRandomSampleNoReplacementMR,self).configure_options()
        self.add_passthrough_option("--sample-size",
                                    type= int,
                                    help = "number of elements in sample")
                                    
    def load_options(self,args):
        super(SimpleRandomSampleNoReplacementMR,self).load_options(args)
        if self.options.sample_size is None:
            self.option_parser.error("You need to specify sample size")
        else:
            self.n = self.options.sample_size
            
    def mapper_rs(self,_,line):
        r = random.randrange(1000000)
        if len(self.pq) < self.n:
            heapq.heappush(self.pq,(r,line))
        else:
            if self.pq[0][0] < r:
               heapq.heappushpop(self.pq,(r,line))
            
    def mapper_rs_final(self):
        yield 1, self.pq
        
    def reducer_rs(self,key,samples):
        pq_final = []
        for sample in samples:
            for element in sample:
                if len(pq_final) < self.n:
                   pq_final.append(element)
                   if len(pq_final)==self.n:
                       heapq.heapify(pq_final)
                else:
                    if pq_final[0][0] < element[0]:
                        heapq.heappushpop(pq_final,element)
        for r,line in pq_final:
            yield None, line
            
    def steps(self):
        return [MRStep(mapper = self.mapper_rs,
                       mapper_final = self.mapper_rs_final,
                       reducer = self.reducer_rs)]
                       
if __name__=="__main__":
    SimpleRandomSampleNoReplacementMR.run()
            
        
            
        
        
        
        
        
                                    