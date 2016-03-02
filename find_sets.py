#core algorithm adapted from http://code.activestate.com/recipes/578508-finding-sets-in-the-card-game-set/
import sys
# bit masks for low and high bits of the attributes        

class Find_Sets(object):

    # defined here: results list
    def __init__(self, inp=None, print_results=False):
        if inp==None:
            self.results=[];
            return
        if type(inp)==dict and all([type(inp[i])==dict for i in inp]):
            # collection of sets given
            results = [self.find_sets(inp[i]) for i in inp]
            if print_results:
                for i in results: print i
            self.results = results
        if type(inp)==dict and all(len(inp[i])==4 for i in inp):
            # single set given
            result = self.find_sets(inp)
            if print_results:
                print i
            self.results = [results]

    mask0 = sum(1<<(2*i) for i in range(4))    # 01010101
    mask1 = sum(1<<(2*i+1) for i in range(4))  # 10101010

    def _find_all_sets(self, bit_cards):  # using thirdcard_fast
        found = []
        have = [None for _ in range(256)]
        cards_list = bit_cards.keys()
        for pos,card in enumerate(cards_list):
            have[card]=pos
        for i,ci in enumerate(cards_list):
            for j,cj in enumerate(cards_list[i+1:],i+1):
                third_card_bits = self._thirdcard_fast(cj, ci)
                k = have[third_card_bits]
                if k > j:  # False if k is None, prevents doubly counting sets
                    found.append((bit_cards[ci],bit_cards[cj],bit_cards[third_card_bits]))
        return found
      
    # which third card is needed to complete the set
    # using the 8-bit representation
    def _thirdcard_fast(self, first, second):
        # NB returns bits
        x, y = first, second
        xor = x^y
        swap = ((xor & self.mask1) >> 1) | ((xor & self.mask0) << 1)
        return (x&y) | (~(x|y) & swap)

    #convert_to_bits: expects a dict "set_table" whose key values are labels of set cards
    #(in our case, the labels will correspond to bounding boxes) and whose value is a tuple
    def _convert_to_bits(self, set_table):
        bits = lambda attrs: sum(a<<(2*i) for i,a in enumerate(attrs[::-1]))
        bit_cards = {bits(attributes): label for attributes, label in set_table.iteritems()}
        return bit_cards

    # find_sets.py: Takes in a SET table and returns all sets within it, returning None if there are no sets. 
    # Expects "set_table" to be a dictionary whose key values are length 4 tuples whose values
    # indicate the particular type of a given quality (color, shape, shading, number) and 
    # whose values are labels (positions) of set cards

    def find_sets(self, set_table, append_results=False, print_results=True):
        bit_cards = self._convert_to_bits(set_table)
        all_sets = self._find_all_sets(bit_cards)
        if print_results:
            print all_sets
        if append_results:
            self.results.extend(all_sets)
        return all_sets

def test():
    ex_set_table = {(2,0,1,0): '1', (0,0,0,1): '2', (0,2,2,0): '3', (1,1,0,0): '4', (0,1,1,2):'5', (1,1,2,0): '6'}
    FS = Find_Sets()
    FS.find_sets(ex_set_table)
    ex_set_table2 = {(0,2,2,2): '1', (2,1,1,2): '2', (1,0,1,2): '3',  (0,1,0,1): '4',\
            (2,2,2,1): '5', (1,2,2,0):'6', (1,1,1,0): '7', (0,0,1,0):'8', (1,2,1,2): '9',\
            (0,0,0,1): '10', (2,2,0,2):'11', (2,2,1,0): '12'}
    FS.find_sets(ex_set_table2)

if __name__ == "__main__":
    test()
                   
