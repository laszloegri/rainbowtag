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


from dynamic_color_master_table import *

def make_LMS_to_Lab_and_IPT():
    m = LMS_quantization_number + 1
    LMS_to_Lab_IPT = np.empty((6, m, m, m), dtype=np.uint8)
    for L_n in range(m):
        for M_n in range(m):
            for S_n in range(m):
                L, M, S = change_0_n_interval_to_LMS(L_n, M_n, S_n)

                # Go back to XYZ
                adapted_xyz_color = convert_bradford_LMS_to_CIEXYZ(L, M, S)

                # CIELAB
                lab_color = convert_color(adapted_xyz_color, LabColor)
                print(lab_color)
                t_lab = lab_color.get_value_tuple()
                Lab_L = np.uint8(np.round(t_lab[0] / 100.0 * 255, 0))
                Lab_a = np.uint8(np.round(t_lab[1] / 100.0 * 127 + 128, 0))
                Lab_b = np.uint8(np.round(t_lab[2] / 100.0 * 127 + 128, 0))

                # IPT
                ipt_color = convert_color(adapted_xyz_color, IPTColor)
                t_ipt = ipt_color.get_value_tuple()
                IPT_I = np.uint8(np.round(t_ipt[0] * 255, 0))
                IPT_P = np.uint8(np.round(t_ipt[1] * 127 + 128, 0))
                IPT_T = np.uint8(np.round(t_ipt[2] * 127 + 128, 0))

                # Set array values
                LMS_to_Lab_IPT[0][L_n, M_n, S_n] = Lab_L
                LMS_to_Lab_IPT[1][L_n, M_n, S_n] = Lab_a
                LMS_to_Lab_IPT[2][L_n, M_n, S_n] = Lab_b
                LMS_to_Lab_IPT[3][L_n, M_n, S_n] = IPT_I
                LMS_to_Lab_IPT[4][L_n, M_n, S_n] = IPT_P
                LMS_to_Lab_IPT[5][L_n, M_n, S_n] = IPT_T


        print("Producing LMS to Lab and IPT table. Progress: " + str(L_n) + " / " + str(LMS_quantization_number))
    # Save results
    if LMS_quantization_number == 255:
        np.save("ColorExplorer_2D_256/LMS_to_Lab_IPT_table.npy", LMS_to_Lab_IPT)
    else:
        np.save("ColorExplorer_2D_512/LMS_to_Lab_IPT_table.npy", LMS_to_Lab_IPT)