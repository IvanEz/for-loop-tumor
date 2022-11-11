#!/bin/sh

path_patient="t1.nii.gz"
path_atlas="../atlas_resized_229to129.nii"
path_inv_mask_f="Tum_Combo_inverted.nii.gz" #mask that excludes tumor (in patient space)
path_tumor_mask_f="Tum_Combo.nii.gz" # is mask from FLAIR scan
path_tumor_mask_t="Tum_T1_binarized.nii.gz" # is mask from T1 scan
path_pet="Tum_FET_necro.nii.gz"

outputfile="morphed_to_atlas"


if test -f "$path_patient"; then
    echo "$path_patient exists."
else
    path_patient="t1c.nii.gz"
fi

ANTS 3 -m CC[$path_patient,$path_atlas,1,4] -i 50x20x0 -o $outputfile -t SyN[0.25] -x $path_inv_mask_f -r Gauss[3,0]


