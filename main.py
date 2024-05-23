import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List

# Constants
MIN_X = 0
MAX_X = 6
MIN_Y = 0
MAX_Y = 6

class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __str__(self):
        return f"[{self.x}, {self.y}]"

    def is_below(self, point) -> bool:
        return point.y <= self.y

class Edge:
    def __init__(self, start: Point, end: Point) -> None:
        self.start = start
        self.end = end
        if self.start.y > self.end.y:
            self.start, self.end = self.end, self.start

    def __str__(self):
        return f"\n{self.start} -> {self.end}"

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

    def display(self, tabulation: str) -> str:
        res = f"\n{tabulation}[LeftRightNode({self.split_edge.start}, {self.split_edge.end}) -> Left: "
        res += self.left.display(tabulation + "\t")
        res += " , Right: "
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

    def display(self, tabulation: str) -> str:
        res = f"\n{tabulation}[UpperLowerNode({self.med}) -> Upper: "
        res += self.upper.display(tabulation + "\t")
        res += " , Lower: "
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
        print(p)
        print("FOUND IN TRAPEZOID:")
        for e in self.edges:
            print(e)

    def display(self, tabulation: str) -> str:
        res = f"\n{tabulation}[Trapezoid -> "
        for e in self.edges:
            res += f"({e.start}, {e.end}) "
        plot_trapezoid(self.edges)
        return res

# Get the median horizontal edge across the vertical range of vertices
def get_med(vertices: List[Point]) -> Edge:
    median = (vertices[-1].y + vertices[0].y) / 2.0
    med_point = min(vertices, key=lambda v: abs(v.y - median))
    return Edge(Point(MIN_X, med_point.y), Point(MAX_X, med_point.y))

# Get the upper horizontal edge based on the highest y-coordinate of vertices
def get_upper_edge(vertices: List[Point]) -> Edge:
    return Edge(Point(MIN_X, vertices[-1].y), Point(MAX_X, vertices[-1].y))

# Get the lower horizontal edge based on the lowest y-coordinate of vertices
def get_lower_edge(vertices: List[Point]) -> Edge:
    return Edge(Point(MIN_X, vertices[0].y), Point(MAX_X, vertices[0].y))

# Get the right vertical edge based on the highest x-coordinate of vertices
def get_right_edge(vertices: List[Point]) -> Edge:
    right_bound = max(vertices, key=lambda v: v.x)
    return Edge(Point(right_bound.x, MIN_Y), Point(right_bound.x, MAX_Y))

# Get the left vertical edge based on the lowest x-coordinate of vertices
def get_left_edge(vertices: List[Point]) -> Edge:
    left_bound = min(vertices, key=lambda v: v.x)
    return Edge(Point(left_bound.x, MIN_Y), Point(left_bound.x, MAX_Y))

# Determine the relative position of point p to edge e (left, right, or on the edge)
def is_left(p: Point, e: Edge) -> int:
    vec1 = Point(e.end.x - e.start.x, e.end.y - e.start.y)
    vec2 = Point(p.x - e.start.x, p.y - e.start.y)
    cross = vec1.x * vec2.y - vec1.y * vec2.x
    if cross > 0:
        return -1  # left
    elif cross < 0:
        return 1  # right
    else:
        return 0

# Check if point p is not to the right of edge e
def is_not_right(p: Point, e: Edge) -> bool:
    return is_left(p, e) != 1

# Decompose the polygon into a root tree structure
def decompose_root(edges: List[Edge], vertices: List[Point], lower_edge: Edge, upper_edge: Edge, right_edge: Edge, left_edge: Edge) -> Root:
    med = get_med(vertices)
    lower = [e for e in edges if e.start.y < med.start.y]
    upper = [e for e in edges if e.end.y > med.start.y]
    lower_v = [v for v in vertices if v.is_below(med.start)]
    upper_v = [v for v in vertices if not v.is_below(med.start)]
    return Root(
        decompose_leaves(lower, lower_v, lower_edge, med, right_edge, left_edge),
        decompose_leaves(upper, upper_v, med, upper_edge, right_edge, left_edge),
        med.start.y
    )

# Decompose the polygon into leaves (trapezoids) based on edges and vertices
def decompose_leaves(edges: List[Edge], vertices: List[Point], lower_edge: Edge, upper_edge: Edge, right_edge: Edge, left_edge: Edge):
    split_edge = None
    inner_point = any(
        (is_not_right(e.start, right_edge) and not is_not_right(e.start, left_edge) and lower_edge.start.y < e.start.y < upper_edge.start.y) or
        (is_not_right(e.end, right_edge) and not is_not_right(e.end, left_edge) and lower_edge.start.y < e.end.y < upper_edge.start.y)
        for e in edges
    )

    for e in edges:
        if e.start.y <= lower_edge.start.y and e.end.y >= upper_edge.start.y:
            split_edge = e
            break

    if split_edge is None:
        if not inner_point:
            return Trapezoid([lower_edge, left_edge, upper_edge, right_edge])
        return decompose_root(edges, vertices, lower_edge, upper_edge, right_edge, left_edge)

    right_edges, right_verts = [], []
    left_edges, left_verts = [], []

    for e in edges:
        if e == split_edge:
            continue
        start_check = is_left(e.start, split_edge)
        end_check = is_left(e.end, split_edge)
        if start_check == -1:
            if e not in left_edges:
                left_edges.append(e)
            if lower_edge.start.y <= e.start.y <= upper_edge.start.y and e.start not in left_verts:
                left_verts.append(e.start)
        elif start_check == 1:
            if e not in right_edges:
                right_edges.append(e)
            if lower_edge.start.y <= e.start.y <= upper_edge.start.y and e.start not in right_verts:
                right_verts.append(e.start)

        if end_check == -1:
            if e not in left_edges:
                left_edges.append(e)
            if lower_edge.start.y <= e.end.y <= upper_edge.start.y and e.end not in left_verts:
                left_verts.append(e.end)
        elif end_check == 1:
            if e not in right_edges:
                right_edges.append(e)
            if lower_edge.start.y <= e.end.y <= upper_edge.start.y and e.end not in right_verts:
                right_verts.append(e.end)

    return Leaf(
        right=decompose_leaves(right_edges, right_verts, lower_edge, upper_edge, split_edge, right_edge),
        left=decompose_leaves(left_edges, left_verts, lower_edge, upper_edge, left_edge, split_edge),
        split_edge=split_edge
    )

# Plot the polygon and points on a plane
def plot_plane(edges: List[Edge], points: List[Point]):
    for e in edges:
        plt.plot([e.start.x, e.end.x], [e.start.y, e.end.y], 'b-')
    for point in points:
        plt.scatter(point.x, point.y, color="red", label="Point")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polygon')
    plt.grid(True)
    plt.xlim(MIN_X - 1, MAX_X)
    plt.ylim(MIN_Y - 1, MAX_Y)

# Plot the trapezoid on a plane
def plot_trapezoid(edges: List[Edge]):
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polygon')
    plt.xlim(MIN_X - 1, MAX_X)
    plt.ylim(MIN_Y - 1, MAX_Y)
    plt.grid(True)
    for e in edges:
        plt.plot([e.start.x, e.end.x], [e.start.y, e.end.y], linestyle='-')

# Sort vertices based on their y-coordinates
def vert_sort(arr: List[Point]):
    arr.sort(key=lambda p: p.y)

# Main function to run the algorithm
def run():
    vertices = [
        Point(0, 0.5), Point(1, 2.5), Point(5.5, 4),
        Point(4, 2), Point(3.5, 0)
    ]
    edges = [
        Edge(vertices[0], vertices[1]), Edge(vertices[1], vertices[2]),
        Edge(vertices[2], vertices[3]), Edge(vertices[3], vertices[4]),
        Edge(vertices[4], vertices[0])
    ]
    points = [Point(random.uniform(MIN_X, MAX_X), random.uniform(MIN_Y, MAX_Y)) for _ in range(5)]
    points.append(Point(0, 1.5))
    plot_plane(edges, points)
    vert_sort(vertices)
    tree = Tree(decompose_root(
        edges, vertices,
        get_lower_edge(vertices), get_upper_edge(vertices),
        get_right_edge(vertices), get_left_edge(vertices)
    ))
    for p in points:
        tree.locate(p)
    tree.display()

run()