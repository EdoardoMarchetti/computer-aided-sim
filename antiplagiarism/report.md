# What are the inputs and the outputs of the software?

The simulator takes as inputs:
- the filepath to the text resource
- the fixed size of the sentences
- the tolerance for the false positives, i.e., the probability
that the system produces a false positive prediction, given an arbitrary sentence.

# How many sentences are stored for S=4 and S=8?

For S=4 95946 unique sentences, while for S=8 96547.

# What is the theoretical amount of stored data in bytes, independently from the adopted data structure?

The theoretical size of the set is between 3.5 and 1.5 bytes, depending on S and the false positive tolerance (see the figure).

# Implement a solution storing the sentence strings in a python set. What is the actual amount of memory occupancy?

4 bytes

# Under which conditions the fingerprinting allows to reduce the actual amount of memory? Is it independent from S? Why?

The fingerprinting allows to reduce the memory occupation only increasing the false positive tolerance.

If we impose a strict tolerance, then we ask few collision, thus more bits are required for storing the fingerprint.

It is independend from S because the size of the fingerprint depend only on the false positive tolerance we impose.

