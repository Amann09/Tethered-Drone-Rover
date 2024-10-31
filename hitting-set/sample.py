import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

tether_length = 3

def eucl_dist(P1, P2):
    (x1,y1) = P1
    (x2,y2) = P2
    return math.sqrt( (x1-x2)**2 + (y1-y2)**2 )

# Function to calculate intersection of two lines defined by points (x1, y1), (x2, y2) and (x3, y3), (x4, y4)
def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Line coefficients
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:  # Lines are parallel or coincident
        return None
    
    # Calculating the intersection point using determinants
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denominator
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denominator
    
    # Check if the intersection point lies on both segments
    if (min(x1, x2) <= px <= max(x1, x2)) and (min(y1, y2) <= py <= max(y1, y2)) and \
       (min(x3, x4) <= px <= max(x3, x4)) and (min(y3, y4) <= py <= max(y3, y4)):
        return (px, py)
    else:
        return None

# Create a 10x10 grid graph
n = 20 # 20
G = nx.grid_2d_graph(n, n)

waypoints = list(G.nodes())

# origin
o = -1
d = n

# Define peripheral nodes 
"""  
d = n-1, o =  0 for nodes to be on the grid 
d = n,   o = -1 for nodes to be 1 unit away from the grid
d = n+1, o = -2 for nodes to be 2 unit away from the grid
d = n-2, o = +1 for nodes to be 1 unit inside the grid 
"""
peripheral_nodes = [(o, d)      , ((o+d)/2, d),     (d, d), 
                    (o, (o+d)/2),                   (d, (o+d)/2),
                    (o, o)      , ((o+d)/2, o),     (d, o)]


# Add peripheral nodes to the graph
for p_node in peripheral_nodes:
    G.add_node(p_node)

# List of green edges (connecting peripheral nodes)
green_edges = [(peripheral_nodes[i], peripheral_nodes[j]) for i in range(len(peripheral_nodes)) for j in range(i + 1, len(peripheral_nodes))]
# print(f"Green Edges length: {len(green_edges)}")
# for edges in green_edges:
#     print(edges, len(green_edges))
#     print("\n")

new_green_edges = []
for edges in green_edges:
    # print(f"current {edges}, {len(green_edges)}")
    x1, y1 = edges[0][0], edges[0][1]
    x2, y2 = edges[1][0], edges[1][1]

    angle = math.atan2((y2-y1),(x2-x1))

    new_green_edges.append(edges + (angle, ))

# Remove horizontal, vertical, or straight diagonal edges
# green_edges = [edge for edge in green_edges if abs(math.atan2(edge[1][1] - edge[0][1], edge[1][0] - edge[0][0])) not in [0, math.pi/2, math.pi]]

newer_green_edges = []
for edges in new_green_edges:
    if abs(edges[2]) not in [float(0), math.pi/2, math.pi]:
        newer_green_edges.append(edges)

# for edges in newer_green_edges:
#     print(edges, len(newer_green_edges))
#     print("\n")

# # Add the green edges to the graph
# for edge in green_edges:
#     G.add_edge(*edge)

# Find intersection points between all green edges
intersection_points = []
for i in range(len(green_edges)):
    for j in range(i + 1, len(green_edges)):
        p1, p2 = green_edges[i]
        p3, p4 = green_edges[j]
        intersection = line_intersection(p1, p2, p3, p4)
        if intersection:
            intersection_points.append(intersection)

# Draw the grid graph
plt.figure(figsize=(200, 200))
pos = dict((n, n) for n in G.nodes())  # Positions as grid coordinates

# Draw grid nodes
nx.draw(G, pos, with_labels=False, node_size=100, node_color='skyblue', font_size=10, font_color='black', alpha=0.35)

# Draw peripheral nodes and green edges
nx.draw_networkx_nodes(G, pos, nodelist=peripheral_nodes, node_size=300, node_color='lightgreen')
nx.draw_networkx_edges(G, pos, edgelist=newer_green_edges, edge_color='green')

# # Mark the intersection points with red dots
# for point in intersection_points:
#     plt.scatter(point[0], point[1], color='violet', s=50)
#     # plt.text(point[0], point[1], f"({point[0]:.2f}, {point[1]:.2f})", fontsize=8, ha='right', color='black')  # Display coordinates


radius = tether_length
for node in G.nodes():
    if node not in peripheral_nodes:
        # circle = plt.Circle((node[0], node[1]), radius, color='purple', linestyle='dotted', fill=False)
        # plt.gca().add_patch(circle)
        (p, q) = (8, 11)
        if (node[0], node[1]) == (p, q):
            circle = plt.Circle((node[0], node[1]), radius, color='purple', linestyle='dotted', fill=False)
            plt.gca().add_patch(circle)



Lines = [[(o, d), (d, (o+d)/2)],
         [(o, d), (d, o)],
         [(o, d), ((o+d)/2, o)],
         [(o, (o+d)/2), ((o+d)/2, d)],
         [(o, (o+d)/2), (d, d)],
         [(o, (o+d)/2), (d, o)],
         [(o, (o+d)/2), ((o+d)/2, o)],
         [(o, o), ((o+d)/2, d)],
         [(o, o), (d, d)],
         [(o, o), (d, (o+d)/2)],
         [((o+d)/2, d), (d, (o+d)/2)],
         [((o+d)/2, o), (d, (o+d)/2)],
         [((o+d)/2, o), (d, d)],
         [((o+d)/2, d), (d, o)]
        ]

length_of_lines = [round(eucl_dist(P[0], P[1]), 3) for P in Lines]

gvPoints = []
for i in range(len(Lines)):
    (x1, y1) = Lines[i][0]
    (x2, y2) = Lines[i][1]
    Dist = length_of_lines[i]
    
    d = 1
    d_vector = ((x2-x1)/Dist, (y2-y1)/Dist)
    points = []
    for n in range(1, round(Dist/d)):
        x_n = x1 + n * d * d_vector[0]
        y_n = y1 + n * d * d_vector[1]
        points.append((x_n, y_n))

    # gvPoints.append(((x1, y1), (x2, y2), points))
    gvPoints.append(points)

# print(gvPoints)
# print("\n")

for points in gvPoints:
    x_coords, y_coords = zip(*points)
    plt.scatter(x_coords, y_coords, color='red', s=40)


plt.scatter([], [], color='red', s=40, label="Ground Vehicles Points")
plt.scatter([], [], color='lightblue', s=100, label="UAV Points")

def calc_eqOfLine_coeff(P1, P2):
    (x1, y1) = P1
    (x2, y2) = P2

    slope = (y2-y1)/(x2-x1)

    a = -slope
    b = 1
    c = - (y1 + (slope*x1))

    return a, b, c


def line_intersects(coeff, center, radius):
    a, b, c = coeff[0], coeff[1], coeff[2]
    x, y = center[0], center[1]

    d = ((abs(a * x + b * y + c)) / math.sqrt(a * a + b * b))
    if radius > d:
        return True
    else:
        return False

s_of_s = []
for waypoint in waypoints:
    (x, y) = waypoint
    intersecting_lines = []
    for line in Lines:
        if line_intersects(coeff=calc_eqOfLine_coeff(line[0], line[1]), center=(x, y), radius=tether_length):
            intersecting_lines.append(line)
    
    s = []
    for line in intersecting_lines:
        idx = Lines.index(line)
        points = gvPoints[idx]
        for point in points:
            (x0, y0)  = point[0], point[1]
            if eucl_dist((x, y), (x0, y0)) < tether_length:
                s.append(point)
    s_of_s.append(s)

index = waypoints.index((p, q))
print(waypoints[index])
print(s_of_s[index], len(s_of_s[index]))
# print(type(s_of_s[0]))


# Show the plot
plt.title("10x10 Grid Graph with Peripheral Nodes and Intersection Points")
plt.legend()
plt.axis('equal')
plt.show()
