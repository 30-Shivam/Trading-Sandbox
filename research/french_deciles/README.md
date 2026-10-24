Survivorship-free factor test using Ken French's free data library (CRSP-based, includes delisted stocks).
Download zips from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/ into ./french/ :
F-F_Research_Data_Factors_CSV, 10_Portfolios_Prior_12_2_CSV, Portfolios_Formed_on_{BE-ME,E-P,AC,OP,INV}_CSV
then `from french_factors import read_block, mkt, stats, excess_t` (see docs/BACKLOG.md for the results).
