#core algorithm adapted from http://code.activestate.com/recipes/578508-finding-sets-in-the-card-game-set/

# bit masks for low and high bits of the attributes        
mask0 = sum(1<<(2*i) for i in range(4))    # 01010101
mask1 = sum(1<<(2*i+1) for i in range(4))  # 10101010

def find_all_sets(bit_cards):  # using thirdcard_fast
    found = []
    have = [None for _ in range(256)]
    cards_list = bit_cards.keys()
    for pos,card in enumerate(cards_list):
    	have[card]=pos
    for i,ci in enumerate(cards_list):
        for j,cj in enumerate(cards_list[i+1:],i+1):
        	third_card_bits = thirdcard_fast(cj, ci)
        	k = have[third_card_bits]
        	if k > j:  # False if k is None, prevents doubly counting sets
        		found.append((bit_cards[ci],bit_cards[cj],bit_cards[third_card_bits]))
	return found
  
# which third card is needed to complete the set
# using the 8-bit representation
def thirdcard_fast(first, second):
    # NB returns bits
    x, y = first, second
    xor = x^y
    swap = ((xor & mask1) >> 1) | ((xor & mask0) << 1)
    return (x&y) | (~(x|y) & swap)

#convert_to_bits: expects a dict "set_table" whose key values are labels of set cards
#(in our case, the labels will correspond to bounding boxes) and whose value is a tuple
def convert_to_bits(set_table):
	bits = lambda attrs: sum(a<<(2*i) for i,a in enumerate(attrs[::-1]))
	bit_cards = {bits(attributes): label for attributes, label in set_table.iteritems()}
	return bit_cards
# find_sets.py: Takes in a SET table and returns all sets within it, returning None if there are no sets. 
# Expects "set_table" to be a dictionary whose key values are length 4 tuples whose values
# indicate the particular type of a given quality (color, shape, shading, number) and 
# whose values are labels (positions) of set cards

def main(set_table):
	bit_cards = convert_to_bits(set_table)
	all_sets = find_all_sets(bit_cards)
	print all_sets
	return all_sets

if __name__ == "__main__":
    main()
               
