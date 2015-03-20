# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

class Max:
    
    def __init__(self, firstIndex, secondIndex, value):
        self.firstIndex = firstIndex
        self.secondIndex = secondIndex
        self.value = value

    def __cmp__(self, other):
        comp =  cmp(abs(self.value), abs(other.value))
        return comp if (comp is not 0) else -cmp(self.secondIndex, other.secondIndex)
