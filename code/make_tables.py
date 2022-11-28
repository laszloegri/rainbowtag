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


from parameters import color_tables_folder,\
    ipt_hue_defs,\
    A_XYZ,\
    B_XYZ,\
    C_XYZ,\
    D50_XYZ,\
    D55_XYZ,\
    D65_XYZ,\
    D75_XYZ,\
    E_XYZ,\
    F2_XYZ,\
    F7_XYZ,\
    F11_XYZ
from dynamic_color_master_table import make_sRGB_to_LMS_table,\
    make_LMS_to_master_table
from color_processing_methods import make_color_master_table
from dynamic_lab_and_ipt_table import make_LMS_to_Lab_and_IPT

# Tables for 2D_color_explorer. This has 256 and 512 version.
# Adjust using LMS_quantization_number in parameters.py
# make_LMS_to_Lab_and_IPT()

# Tables for dynamic chromatic adaptation. make_LMS_to_master_table
# has 256 and 512 version. Adjust using LMS_quantization_number in parameters.py
# make_sRGB_to_LMS_table(color_tables_folder)
# make_LMS_to_master_table(color_tables_folder)

# Precomputed tables for different CIE illuminants
make_color_master_table(A_XYZ, ipt_hue_defs, color_tables_folder, 'a')
make_color_master_table(B_XYZ, ipt_hue_defs, color_tables_folder, 'b')
make_color_master_table(C_XYZ, ipt_hue_defs, color_tables_folder, 'c')
make_color_master_table(D50_XYZ, ipt_hue_defs, color_tables_folder, 'd50')
make_color_master_table(D55_XYZ, ipt_hue_defs, color_tables_folder, 'd55')
make_color_master_table(D65_XYZ, ipt_hue_defs, color_tables_folder, 'd65')
make_color_master_table(D75_XYZ, ipt_hue_defs, color_tables_folder, 'd75')
make_color_master_table(E_XYZ, ipt_hue_defs, color_tables_folder, 'e')
make_color_master_table(F2_XYZ, ipt_hue_defs, color_tables_folder, 'f2')
make_color_master_table(F7_XYZ, ipt_hue_defs, color_tables_folder, 'f7')
make_color_master_table(F11_XYZ, ipt_hue_defs, color_tables_folder, 'f11')