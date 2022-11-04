# !/usr/bin/bash

# goku
echo -e "Validate mesh goku. \n"
python exp_runner.py --mode validate_mesh --conf confs/insect_use_white_bkgd_wmask.conf --case goku_1_1_2_randomview/wobox -r 512 

# shiba
echo -e "Validate mesh shiba. \n"
python exp_runner.py --mode validate_mesh --conf confs/insect_use_white_bkgd_wmask.conf --case shiba_1_2_2_randomview/wobox -r 512

# insect2
echo -e "Validate mesh insect2. \n"
python exp_runner.py --mode validate_mesh --conf confs/insect_use_white_bkgd_wmask.conf --case insect2_2_2_1_randomview/wobox -r 512

# dinosaur
echo -e "Validate mesh dinosaur. \n"
python exp_runner.py --mode validate_mesh --conf confs/insect_use_white_bkgd_wmask.conf --case dinosaur_2_1_14_randomview/wobox -r 512

echo "Done!"