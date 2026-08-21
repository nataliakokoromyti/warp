#!/usr/bin/env python3
"""Replace the dynamic vector index in mujoco_warp's octree descent with selects.

``find_oct`` picks the next octree child with ``oct_child[node][4*z + 2*y + x]``.
That is a *dynamic* index into an 8-wide value held in registers. AMD GPUs have
no indexed register file access, so the compiler either spills the vector to
scratch or emits a select chain; inside a 100-iteration pointer-chasing loop
that is the hottest thing in the kernel. Writing the select chain explicitly
keeps every index static.
"""

import sys

PATH = sys.argv[1]
src = open(PATH).read()

OLD = """    x = 0 if coord[0] < 0.5 else 1
    y = 0 if coord[1] < 0.5 else 1
    z = 0 if coord[2] < 0.5 else 1
    child = oct_child[node][4 * z + 2 * y + x]"""

NEW = """    x = 0 if coord[0] < 0.5 else 1
    y = 0 if coord[1] < 0.5 else 1
    z = 0 if coord[2] < 0.5 else 1
    # Static-index the child vector. A dynamic index into an 8-wide register
    # value has no hardware support on AMD GPUs and spills it to scratch inside
    # this hot descent loop; an explicit select chain keeps it in registers.
    octant = 4 * z + 2 * y + x
    children = oct_child[node]
    child = children[0]
    child = wp.where(octant == 1, children[1], child)
    child = wp.where(octant == 2, children[2], child)
    child = wp.where(octant == 3, children[3], child)
    child = wp.where(octant == 4, children[4], child)
    child = wp.where(octant == 5, children[5], child)
    child = wp.where(octant == 6, children[6], child)
    child = wp.where(octant == 7, children[7], child)"""

assert OLD in src, "octree descent anchor missing"
src = src.replace(OLD, NEW, 1)

# The leaf test re-reads oct_child[node] eight times; hoist it to one load.
OLD2 = """    child0 = oct_child[node][0]
    # Evaluate this hot leaf predicate eagerly to avoid branch-heavy codegen.
    if (
      int(child0 == -1)
      & int(oct_child[node][1] == -1)
      & int(oct_child[node][2] == -1)
      & int(oct_child[node][3] == -1)
      & int(oct_child[node][4] == -1)
      & int(oct_child[node][5] == -1)
      & int(oct_child[node][6] == -1)
      & int(oct_child[node][7] == -1)
    ) != 0:"""
NEW2 = """    node_children = oct_child[node]
    # Evaluate this hot leaf predicate eagerly to avoid branch-heavy codegen.
    if (
      int(node_children[0] == -1)
      & int(node_children[1] == -1)
      & int(node_children[2] == -1)
      & int(node_children[3] == -1)
      & int(node_children[4] == -1)
      & int(node_children[5] == -1)
      & int(node_children[6] == -1)
      & int(node_children[7] == -1)
    ) != 0:"""
if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1)
    print("hoisted leaf-test load")
else:
    print("WARNING: leaf-test anchor not found, skipped")

open(PATH, "w").write(src)
print("patched octree descent in", PATH)
