# -*- coding: utf-8 -*-
"""
#########################################################################

 University of Rostock
 Faculty of Informatics and Electrical Engineering
 Institute of Communications Engineering

#########################################################################

This class creates a Mapper for ASK,PSK and M-QAM mapping

M                   - modulation index

modulation          - modulation scheme ('ASK'/'PSK'/'QAM'}
                         
mapping             - mapping scheme:
                        == 'gray'               : gray mapping
                        == 'antigray'           : anti gray mapping
                        == 'natural'/'binary'   : natural mapping
                        == 'rand'               : random mapping
                         (default == 'gray')
                         
init_phase          - phase rotation of symbols
                        (default == 0)

normalize_energy    - flag to define if symbol energy is normalized
                        (default == False)
-------------------------------------------------------------------------
Example:
    u = [1,0,1,0,1,0,1,0]
    mapper = map.Mapper()
    mapper.generate_mapping(4,'ASK','gray',0)
    mapper.show_mapping()
    x = mapper.map_symbol(u)
    y = mapper.demap_symbol_hard(x[0],1)

@author: Steffen Steiner
"""
import numpy as np
import matplotlib.pyplot as plt

class Mapper:
    
    
    def __init__(self):
        self.mapping = None
                
    def generate_mapping(self, M, modulation, mapping=None, init_phase=None, normalize_energy=None):
        """Creates the mapping scheme for further use."""
        if mapping == None:
            mapping = "gray" # no mapping
        if init_phase == None:
            init_phase = 0 #no init phase
        if normalize_energy == None:
            normalize_energy = False #no normalization
        
        self.mapping = self.Mapping(M, modulation, mapping, init_phase,normalize_energy)
    
    
    
    def map_symbol(self,c):
        """Maps the given bits in c to symbols with the predefined mapping scheme."""
        assert self.mapping!=None, "No mapping scheme defined!"
        
        #zero padding
        padded_bits = np.ceil(np.size(c)/self.mapping.m) * self.mapping.m - np.size(c);
        if padded_bits > 0:
            c = np.asarray(np.hstack((c, np.zeros(1,padded_bits))),dtype=np.int)
        
        c = str(c.reshape(self.mapping.m,int(np.size(c)/self.mapping.m)))
        c =  c.replace("[", ""); c =  c.replace(" ", ""); c =  c.replace("]", "")
        c = c.split('\n')
        c = [int(x,2) for x in (''.join(y) for y in list(zip(*c)))]
        #do mapping
        x = self.mapping.alphabet_complex[c];
        x_bin = self.mapping.alphabet_binary[c];
        return(x,x_bin)        

        
        
    def demap_symbol_hard(self,y, padded_bits=None,h_hat = None):
        """Demaps the given symbols in y to bits with the predefined mapping scheme."""
        assert self.mapping!=None, "No mapping scheme defined!"
        
        if padded_bits == None:
            padded_bits = 0 # no zero padding
        assert padded_bits >= 0, "The number of padded bits need to be positive!"
        
        if h_hat == None:
            h_hat = 1
        assert np.isscalar(h_hat), "The estimation of the channel coefficient must be a scalar value!"
        
        
        #Hard Decision -> calculate distance between distrubed alphabet and received symbols
        dist = np.abs(np.outer(np.ones(np.size(y)), self.mapping.alphabet_complex) * h_hat - np.outer(y, np.ones(np.power(2,self.mapping.m))));

        # find minimum distance
        min_idx = [np.argmin(x) for x in dist]

        #choose binary digits from alphabet
        c_hat = self.mapping.alphabet_binary[min_idx]
        
        #remove bits added by zero padding
        tmp = np.array2string(c_hat)
        tmp =  tmp.replace("[", ""); tmp =  tmp.replace(" ", ""); tmp =  tmp.replace("]", ""); tmp = tmp.replace("'","")
        tmp = tmp[0:len(tmp)-1-padded_bits]
        c_hat = np.array([x for x in tmp])
             
        return c_hat
    
    
    
    def demap_symbol_soft(self,y, padded_bits, EsNo_dB,  Lc_a, h_hat, algorithm):
        """Demaps the given symbols in y to bits with the predefined mapping scheme."""
        assert self.mapping!=None, "No mapping scheme defined!"
        
        #TODO implement
        return
   
    
    
    def show_mapping(self):
        """Plots the predefined mapping scheme."""
        
        plt.scatter(np.real(self.mapping.alphabet_complex), np.imag(self.mapping.alphabet_complex))
        plt.title('Mapping scheme')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid('on')

        if self.mapping.M <= 64:
            for k in np.arange(0,self.mapping.M):
                #decimal
                label_dec = str(self.mapping.alphabet_decimal[k]);
                plt.annotate(label_dec,(np.real(self.mapping.alphabet_complex[k]), np.imag(self.mapping.alphabet_complex[k])+0.1), ha='center')
                #binary
                label_bin = str(self.mapping.alphabet_binary[k]);
                plt.annotate(label_bin,(np.real(self.mapping.alphabet_complex[k]), np.imag(self.mapping.alphabet_complex[k])-0.3), ha='center')
                
        plt.show()
        
    def get_mapping_description(self):
        return self.mapping.description

    class Mapping:
        
        def __init__(self, M, modulation, mapping, init_phase, normalize_energy):
            self.M = M
            self.modulation = modulation
            self.mapping = mapping
            self.init_phase = init_phase
            self.normalize_energy = normalize_energy
            self.m = int(np.log2(M))
            self.description = str(M) + '-' + modulation+' (' + mapping + ')'
            self.create_alphabets()

        def create_alphabets(self):           
            self.alphabet_decimal = np.arange(0,int(np.power(2,self.m)))
            self.alphabet_binary = np.array([np.binary_repr(x,self.m) for x in self.alphabet_decimal])
         
            complex = np.zeros(self.M)
            if self.modulation.upper() == "ASK":
                complex =np.arange(1,int(np.power(2,self.m))+1,dtype=float)
                complex = complex - np.mean(complex)
            elif self.modulation.upper() == "PSK":
                complex = [np.exp(1j * 2*np.pi*(x/self.M)) for x in np.arange(0,self.M)]
            elif self.modulation.upper() == "QAM":
                assert np.mod(np.log(self.M)/np.log(4),1) == 0, "For QAM the modulation index M need to be a positive power of four! (4,16,64,...)"
                height = np.sqrt(self.M)
                [Re, Im] = np.meshgrid(np.arange(1,height+1),np.arange(height,0,-1))
                Re = Re - np.mean(Re[:])
                Im = Im - np.mean(Im[:])
                complex = np.transpose(Re + 1j*Im)
                complex = complex.flatten()
            else:
                raise Exception("Unsupported modulation scheme!")
            self.alphabet_complex = complex
            
            
            #Normalize symbol-energy
            if self.normalize_energy == True:
                P = np.sum(np.power(np.abs(self.alphabet_complex),2))/self.M
                self.alphabet_complex =self.alphabet_complex / np.sqrt(P);
 
            #Determine mapping
            if self.mapping.upper() == "NATURAL" or self.mapping.upper() == "BINARY":
                perm = np.arange(1,np.power(2,self.m)+1)
                perm = np.argsort(perm)
                
            elif self.mapping.upper() == "RAND" or self.mapping.upper() == "RANDOM":
                perm =  np.random.permutation(np.power(2,self.m)) 
                perm = np.argsort(perm)
                
            elif self.mapping.upper() == "GRAY":
                if self.modulation.upper() != "QAM":
                    perm = np.arange(0,np.power(2,self.m))
                    perm = [int(x,2) for x in self.dec2gray(perm)]
                    perm = np.argsort(perm)
                    
                else:
                   perm = np.array([[0,2], [1,3]])
                   m_tmp = 2;
                   while m_tmp < self.m:
                       perm = np.array([[perm+0*np.power(2,m_tmp), np.fliplr(perm)+2*np.power(2,m_tmp)],[np.flipud(perm)+1*np.power(2,m_tmp), np.flipud(np.fliplr(perm))+3*np.power(2,m_tmp)]])
                       m_tmp = m_tmp + 2
                   perm = perm.flatten(1)
                   perm = np.argsort(perm)
                         
            elif self.mapping.upper() == "ANTIGRAY":
                if self.modulation.upper() != "QAM":
                     perm = self.antigray(np.power(2,self.m))
                     perm = perm.astype(int)
                     perm = str(perm)
                     perm =  perm.replace("[", "")
                     perm =  perm.replace(" ", "")
                     perm =  perm.replace("]", "")
                     perm = perm.split('\n')
                     perm = [int(x,2) for x in perm]
                     perm = np.argsort(perm)
      
                else:
                    # generated with search_QAM_antigray.m -> function has to be implemented in python
                    if self.M == 4:
                        perm = [4,3,1,2]
                    elif self.M == 16:
                        perm = [8,3,13,10,6,1,15,12,11,16,2,5,9,14,4,7]
                    elif self.M == 64:
                        perm = [
                            38,    31,     3,    29,   47,    22,    40,    13,    19,    45,    41,    61,    54,     6,    50,    56,    16,
                            51,     7,    27,    60,     9,    44,    18,    12,    36,    23,    34,    32,    63,    57,    25,    64,    49,
                             1,    55,    33,     4,    35,    20,    17,    43,    10,    52,    26,    15,    59,    24,    62,    58,    14,
                            46,    53,    42,    37,    11,    21,    48,     5,    39,    28,     2,    30,     8
                            ]
                    else:
                        raise Exception("Unsupported modulation index for QAM and antigray mapping!")
            else:
                 raise Exception("Unsupported mapping scheme!")

            # shift by permutation
            self.alphabet_complex =  self.alphabet_complex[perm]
            
            # phase rotation
            self.alphabet_complex =  self.alphabet_complex * np.exp(1j * self.init_phase);

        #---------------------- Helper functions-------------------------------
        def bin2gray(self,b):
            result = []
            for j in np.arange(0,np.size(b)):
                g = b[j][0]
                for i in np.arange(1,len(b[j])):
                    x = int(b[j][i-1])^int(b[j][i])
                    g = g+str(x)
                result.append(g)
            return result
            
         
            
        def dec2gray(self,d):
            b = [np.binary_repr(x,self.m) for x in d]
            g = self.bin2gray(b);
            return g
          
            
        
        def antigray(self,len):
            assert np.mod(len,2)==0, "The lenght must be a power of two!"
  
            if len == 2:
                antigray_bins = [False, True]
            elif len > 2:
                antigray_bins = self.antigray(len / 2);
                
                tmp1 = np.full((1,np.size(antigray_bins,0)), True, dtype=bool)
                tmp0 = np.full((1,np.size(antigray_bins,0)), False, dtype=bool)
                tmp = np.concatenate((tmp0,tmp1), axis=0)
                tmp = tmp.flatten(1)
        
                antigray_bins = np.concatenate((antigray_bins,list(~np.array(antigray_bins))), axis=0) 
                antigray_bins = np.c_[tmp,antigray_bins]
                
            return antigray_bins
  
    
    
        def search_QAM_antigray(self,M):
            #TODO implement
            return
                
                

