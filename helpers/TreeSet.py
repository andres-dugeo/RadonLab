# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
from bintrees import BinaryTree


class TreeSet:

    def __init__(self):
        self._binaryTree = BinaryTree()
        self._dictionary = {}

    def add(self, object):
        self._binaryTree.insert(object, None)
        self._dictionary.update({object.secondIndex: object})

    def remove(self, object):
        if(object.secondIndex in self._dictionary):
            object = self._dictionary.pop(object.secondIndex)
            self._binaryTree.remove(object)
            return True
        return False

    def pollLast(self):
        if self._binaryTree.is_empty():
            return None
        max = self._binaryTree.pop_max()[0]
        self._dictionary.pop(max.secondIndex)
        return max
