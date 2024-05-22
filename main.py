import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List

MIN_X = 0
MAX_X = 7
MIN_Y = 0
MAX_Y = 7

class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def print(self):
        print("[", self.x, ", ", self.y, "]")

    def is_below(self, point):
        if point.y <= self.y:
            return True
        return False

class Edge:
    def __init__(self, start: Point, end: Point) -> None:
        self.start = start
        self.end = end
        if self.start.y > self.end.y:
            self.start, self.end = self.end, self.start

    def print(self):
        print("\n")
        self.start.print()
        print(" -> ")
        self.end.print()


class Leaf:
    def __init__(self, left, right, split_edge: Edge) -> None:
        self.left = left
        self.right = right
        self.split_edge = split_edge

    def locate(self, p: Point):
        if is_not_right(p, self.split_edge):
            self.left.locate(p)
        else:
            self.right.locate(p)

    def display(self, tabulation):
        res = "\n" + tabulation  + "[LeftRightNode([" + str(self.split_edge.start.x) + "," + str(self.split_edge.start.y) + "],[" + str(self.split_edge.end.x) + "," + str(self.split_edge.end.y)  + "]) -> Left : "
        res += self.left.display(tabulation + "\t")
        res += " , Right : "
        res += self.right.display(tabulation + "\t")
        res += "]"
        return res


class Root:
    def __init__(self, lower: Leaf, upper: Leaf, med: int) -> None:
        self.lower = lower
        self.upper = upper
        self.med = med

    def locate(self, p: Point):
        if p.y >= self.med:
            self.upper.locate(p)
        else:
            self.lower.locate(p)

    def display(self, tabulation):
        res = "\n" + tabulation + "[UpperLowerNode(" + str(self.med) + ") -> Upper : "
        res += self.upper.display(tabulation + "\t")
        res += " , Lower : "
        res += self.lower.display(tabulation + "\t")
        res += "]"
        return res


class Tree:
    def __init__(self, root: Root) -> None:
        self.root = root

    def locate(self, p: Point):
        self.root.locate(p)

    def display(self):
        tabulation = "\t"
        res = "TREE:\n"
        res += self.root.display(tabulation)
        print(res)
        plt.show()

class Trapezoid:
    def __init__(self, edges: List[Edge]) -> None:
        self.edges = edges

    def locate(self, p: Point):
        print("POINT:")
        p.print()
        print("FOUND IN TRAPEZOID:")
        for e in self.edges:
            e.print()

    def display(self, tabulation):
        res = "\n" + tabulation + "[Trapezoid -> "
        for e in self.edges:
            res += "([" + str(e.start.x) + "," + str(e.start.y) + "],[" + str(e.end.x) + "," + str(e.end.y)  + "]) "
        plot_trapezoid(self.edges)
        return res
        #plt.show()