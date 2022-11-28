"""
This is a Python implementation of the RainbowTag fiducial marker published here:
https://ieeexplore.ieee.org/document/9743123

The project was funded by NSERC and Indus.ai (acquired by Procore https://www.procore.com/en-ca)
through Concordia University.

This code can be used for research purposes. For other purposes, 
please check with Concordia University, Indus.ai and NSERC.

Code author: Laszlo Egri
"""

__author__ = "Laszlo Egri"
__copyright__ = "Copyright 2019, Concordia University, Indus.ai, NSERC"


import numba as nb
import numpy as np
from numba.typed import List
import cv2


@nb.jit
def get_edges_from_adjacency_matrix(adjacency_matrix):
    adjacency_matrix = adjacency_matrix.astype(np.uint8)
    size = adjacency_matrix.shape[0]
    pairs = List()

    for col in nb.prange(size):
        for row in nb.prange(size):
            if row >= col:
                continue
            if adjacency_matrix[row, col]:
                pairs.append((row, col))
    return(pairs)


@nb.njit
def repeatedly_remove_low_degree_vertices_from_adjacency_table(adjacency_matrix):
    size = adjacency_matrix.shape[0]
    change = True
    while change:
        change = False
        for col in nb.prange(size):
            degree = 0
            small_degree = True
            for row in range(size):
                if adjacency_matrix[col, row]:
                    degree = degree + 1
                if degree == 4:
                    small_degree = False
                    break
            if small_degree and (degree > 0):
                change = True
                for row in range(size):
                    adjacency_matrix[col, row] = False
                    adjacency_matrix[row, col] = False
    return(adjacency_matrix)


@nb.jit
def remove_isolated_vertices(adjacency_matrix, blobs_with_stats):
    size = adjacency_matrix.shape[0]
    # It is important to initialize to False
    row_reduced_adjacency_matrix = np.zeros((size, size), nb.boolean)
    reduced_blobs_with_stats = np.empty((size, 7), dtype=nb.int32)

    # First remove all zero columns
    reduced_row = 0
    for row in range(size):
        if np.any(adjacency_matrix[row]):
            row_reduced_adjacency_matrix[reduced_row] = adjacency_matrix[row]
            reduced_blobs_with_stats[reduced_row] = blobs_with_stats[row]
            reduced_row = reduced_row + 1

    # Cut off garbage end
    reduced_blobs_with_stats = reduced_blobs_with_stats[0:reduced_row]

    # Then remove all zero rows. (Matrix is symmetric so we could optimize
    # but this won't speed up things much so leave it.)
    reduced_adjacency_matrix = np.zeros((size, size), nb.boolean)
    reduced_col = 0
    for col in range(size):
        if np.any(row_reduced_adjacency_matrix[:,col]):
            reduced_adjacency_matrix[reduced_col] = row_reduced_adjacency_matrix[:,col]
            reduced_col = reduced_col + 1

    reduced_adjacency_matrix = reduced_adjacency_matrix[0:reduced_row, 0:reduced_col]
    return(reduced_adjacency_matrix, reduced_blobs_with_stats)


# This method reorders the rows and columns of adjacency_matrix
# into block of the same color.
@nb.jit
def reorder_rows_and_columns_to_RMYGB(adjacency_matrix, blobs_with_stats):
    size = adjacency_matrix.shape[0]

    R_block = np.empty((size, size), nb.boolean)
    r_counter = 0

    M_block = np.empty((size, size), nb.boolean)
    m_counter = 0

    Y_block = np.empty((size, size), nb.boolean)
    y_counter = 0

    G_block = np.empty((size, size), nb.boolean)
    g_counter = 0

    B_block = np.empty((size, size), nb.boolean)
    b_counter = 0

    row_reordered_adjacency_matrix = np.zeros((size, size), nb.boolean)
    reordered_adjacency_matrix = np.zeros((size, size), nb.boolean)

    reordered_blobs_with_stats_R = np.empty((size, 7), dtype=nb.int32)
    reordered_blobs_with_stats_M = np.empty((size, 7), dtype=nb.int32)
    reordered_blobs_with_stats_Y = np.empty((size, 7), dtype=nb.int32)
    reordered_blobs_with_stats_G = np.empty((size, 7), dtype=nb.int32)
    reordered_blobs_with_stats_B = np.empty((size, 7), dtype=nb.int32)
    reordered_blobs_with_stats = np.empty((size, 7), dtype=nb.int32)

    # REORDER ROWS FIRST
    for row in range(size):
        color_code = blobs_with_stats[row][3]
        if color_code == 1:
            R_block[r_counter] = adjacency_matrix[row]
            reordered_blobs_with_stats_R[r_counter] = blobs_with_stats[row]
            r_counter = r_counter + 1
        if color_code == 5:
            M_block[m_counter] = adjacency_matrix[row]
            reordered_blobs_with_stats_M[m_counter] = blobs_with_stats[row]
            m_counter = m_counter + 1
        if color_code == 2:
            Y_block[y_counter] = adjacency_matrix[row]
            reordered_blobs_with_stats_Y[y_counter] = blobs_with_stats[row]
            y_counter = y_counter + 1
        if color_code == 3:
            G_block[g_counter] = adjacency_matrix[row]
            reordered_blobs_with_stats_G[g_counter] = blobs_with_stats[row]
            g_counter = g_counter + 1
        if color_code == 4:
            B_block[b_counter] = adjacency_matrix[row]
            reordered_blobs_with_stats_B[b_counter] = blobs_with_stats[row]
            b_counter = b_counter + 1

    rm = r_counter + m_counter
    rmy = rm + y_counter
    rmyg = rmy + g_counter
    rmygb = rmyg + b_counter

    row_reordered_adjacency_matrix[0:r_counter] = R_block[0:r_counter]
    row_reordered_adjacency_matrix[r_counter:rm] = M_block[0:m_counter]
    row_reordered_adjacency_matrix[rm:rmy] = Y_block[0:y_counter]
    row_reordered_adjacency_matrix[rmy:rmyg] = G_block[0:g_counter]
    row_reordered_adjacency_matrix[rmyg:rmygb] = B_block[0:b_counter]

    reordered_blobs_with_stats[0:r_counter] = reordered_blobs_with_stats_R[0:r_counter]
    reordered_blobs_with_stats[r_counter:rm] = reordered_blobs_with_stats_M[0:m_counter]
    reordered_blobs_with_stats[rm:rmy] = reordered_blobs_with_stats_Y[0:y_counter]
    reordered_blobs_with_stats[rmy:rmyg] = reordered_blobs_with_stats_G[0:g_counter]
    reordered_blobs_with_stats[rmyg:rmygb] = reordered_blobs_with_stats_B[0:b_counter]


    # THEN REORDER COLUMNS
    r_counter = 0
    m_counter = 0
    y_counter = 0
    g_counter = 0
    b_counter = 0
    for col in range(size):
        color_code = blobs_with_stats[col][3]
        if color_code == 1:
            R_block[:,r_counter] = row_reordered_adjacency_matrix[:,col]
            r_counter = r_counter + 1
        if color_code == 5:
            M_block[:,m_counter] = row_reordered_adjacency_matrix[:,col]
            m_counter = m_counter + 1
        if color_code == 2:
            Y_block[:,y_counter] = row_reordered_adjacency_matrix[:,col]
            y_counter = y_counter + 1
        if color_code == 3:
            G_block[:,g_counter] = row_reordered_adjacency_matrix[:,col]
            g_counter = g_counter + 1
        if color_code == 4:
            B_block[:,b_counter] = row_reordered_adjacency_matrix[:,col]
            b_counter = b_counter + 1

    reordered_adjacency_matrix[:,0:r_counter] = R_block[:,0:r_counter]
    reordered_adjacency_matrix[:,r_counter:rm] = M_block[:,0:m_counter]
    reordered_adjacency_matrix[:,rm:rmy] = Y_block[:,0:y_counter]
    reordered_adjacency_matrix[:,rmy:rmyg] = G_block[:,0:g_counter]
    reordered_adjacency_matrix[:,rmyg:rmygb] = B_block[:,0:b_counter]

    # Red and magenta are together
    color_class_sizes = np.array([r_counter + m_counter, y_counter, g_counter, b_counter])
    color_class_starting_indices = np.array([0,rm,rmy,rmyg,rmygb])
    return(reordered_adjacency_matrix,
           reordered_blobs_with_stats,
           color_class_sizes,
           color_class_starting_indices)


# NOT USED
@nb.jit(cache=True)
def find_4_color_cliques_in_ordered_adjacency_graph(reordered_adjacency_matrix,
                                                    reordered_blobs_with_stats,
                                                    color_class_sizes,
                                                    color_class_starting_indices):

    # Find the two colors which have the least number of blobs
    # Red and magenta are together.
    # Then we compute the row indices of the adjacency matrix for
    # these two sets.

    # blob_set_size_order = np.argsort(color_class_sizes)
    # first_color = blob_set_size_order[0] #0:RM, 1:Y, 2:G, 3:B

    # FOR NOW, WE ALWAYS START WITH RM GROUP AS ANY STICKER MUST HAVE AT LEAST
    # ONE BLOB IN THIS SET.
    first_color = 0

    first_start_index = color_class_starting_indices[first_color]
    first_finish_index = color_class_starting_indices[first_color + 1]

    # Intervals (used with range function, so first index is included, last is excluded)
    rm_start = color_class_starting_indices[0]
    rm_finish = color_class_starting_indices[1]
    y_start = color_class_starting_indices[1]
    y_finish = color_class_starting_indices[2]
    g_start = color_class_starting_indices[2]
    g_finish = color_class_starting_indices[3]
    b_start = color_class_starting_indices[3]
    b_finish = color_class_starting_indices[4]

    four_clique_list = List()

    for first_col in range(first_start_index, first_finish_index):

        first_blob = reordered_blobs_with_stats[first_col]

        if first_color == 0:
        # If first color is 0, i.e., RM, then we have to choose 3 other colors from RM,Y,G,B
        # W enumerate all other cases: {RM,Y,G}, {RM,Y,B}, {RM,G,B}, {Y,G,B}
        # RM below starts at first_col + 1 so that we choose
        # every pair (a,b) \in RM only once.
            cases = np.array([[[first_col + 1, rm_finish], [y_start, y_finish], [g_start, g_finish]],
                              [[first_col + 1, rm_finish], [y_start, y_finish], [b_start, b_finish]],
                              [[first_col + 1, rm_finish], [g_start, g_finish], [b_start, b_finish]],
                              [[y_start, y_finish], [g_start, g_finish], [b_start, b_finish]]])

            for case in range(cases.shape[0]):
                for index_2 in range(cases[case][0][0], cases[case][0][1]):
                    # Is second blob connected to first blob?
                    if reordered_adjacency_matrix[index_2, first_col]:
                        second_blob = reordered_blobs_with_stats[index_2]
                    else:
                        continue

                    # Is third blob connected to first two blobs?
                    for index_3 in range(cases[case][1][0], cases[case][1][1]):
                        # Check RM1-Y and RM2-Y edges.
                        if reordered_adjacency_matrix[index_3, first_col] and \
                            reordered_adjacency_matrix[index_3, index_2]:
                            third_blob = reordered_blobs_with_stats[index_3]
                        else:
                            continue

                        # Is fourth blob connected to first three blobs?
                        for index_4 in range(cases[case][2][0], cases[case][2][1]):
                            # Check RM1-G, RM2-G and Y-G edges.
                            if reordered_adjacency_matrix[index_4, first_col] and \
                                    reordered_adjacency_matrix[index_4, index_2] and \
                                    reordered_adjacency_matrix[index_4, index_3]:
                                fourth_blob = reordered_blobs_with_stats[index_4]
                                four_clique = np.stack((first_blob, second_blob, third_blob, fourth_blob))
                                four_clique_list.append(four_clique)

    return(four_clique_list)


@nb.jit(cache=True)
def find_5_color_cliques_in_ordered_adjacency_graph(reordered_adjacency_matrix,
                                                    reordered_blobs_with_stats,
                                                    color_class_starting_indices):

    # Intervals (used with range function, so first index is included, last is excluded)
    rm_start = color_class_starting_indices[0]
    rm_finish = color_class_starting_indices[1]
    y_start = color_class_starting_indices[1]
    y_finish = color_class_starting_indices[2]
    g_start = color_class_starting_indices[2]
    g_finish = color_class_starting_indices[3]
    b_start = color_class_starting_indices[3]
    b_finish = color_class_starting_indices[4]

    five_clique_list = List()

    # FIRST: RM1
    for rm1_index in range(rm_start, rm_finish):
        rm1_blob = reordered_blobs_with_stats[rm1_index]

        # SECOND: RM2
        for rm2_index in range(rm1_index + 1, rm_finish):

            # Check if rm1_blob and rm2_blob are neighbors
            if reordered_adjacency_matrix[rm1_index, rm2_index]:
                rm2_blob = reordered_blobs_with_stats[rm2_index]
            else:
                continue

            # THIRD: YELLOW
            for y_index in range(y_start, y_finish):
                if reordered_adjacency_matrix[rm1_index, y_index] and \
                        reordered_adjacency_matrix[rm2_index, y_index]:
                    y_blob = reordered_blobs_with_stats[y_index]
                else:
                    continue

                # FOURTH: GREEN
                for g_index in range(g_start, g_finish):
                    if reordered_adjacency_matrix[rm1_index, g_index] and \
                            reordered_adjacency_matrix[rm2_index, g_index] and \
                            reordered_adjacency_matrix[y_index, g_index]:
                        g_blob = reordered_blobs_with_stats[g_index]
                    else:
                        continue

                    # FIFTH: BLUE
                    for b_index in range(b_start, b_finish):
                        if reordered_adjacency_matrix[rm1_index, b_index] and \
                                reordered_adjacency_matrix[rm2_index, b_index] and \
                                reordered_adjacency_matrix[y_index, b_index] and \
                                reordered_adjacency_matrix[g_index, b_index]:
                            b_blob = reordered_blobs_with_stats[b_index]

                            five_clique = np.stack((rm1_blob,
                                                    rm2_blob,
                                                    y_blob,
                                                    g_blob,
                                                    b_blob))
                            five_clique_list.append(five_clique)
    return(five_clique_list)