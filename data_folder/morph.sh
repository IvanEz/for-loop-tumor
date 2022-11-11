#!/bin/sh

path_patient="t1.nii.gz"
path_atlas="../atlas_resized_229to129.nii"
path_inv_mask_f="Tum_Combo_inverted.nii.gz" #mask that excludes tumor (in patient space)
path_tumor_mask_f="Tum_Combo.nii.gz" # is mask from FLAIR scan
path_tumor_mask_t="Tum_T1_binarized.nii.gz" # is mask from T1 scan
#path_pet="Tum_FET_necro.nii.gz"

outputfile="morphed_to_atlas"


if test -f "$path_patient"; then
	echo "$path_patient exists."
else 
	path_patient="t1c.nii.gz"
fi


#inverse -> patient is transformed into atlas space
WarpImageMultiTransform 3 $path_patient patient_to_atlas.nii -R $path_atlas -i morphed_to_atlasAffine.txt morphed_to_atlasInverseWarp.nii.gz
# tumor mask is transformed into atlas space
WarpImageMultiTransform 3 $path_tumor_mask_f tumor_mask_f_to_atlas.nii --use-NN -R $path_atlas -i morphed_to_atlasAffine.txt morphed_to_atlasInverseWarp.nii.gz
WarpImageMultiTransform 3 $path_tumor_mask_t tumor_mask_t_to_atlas.nii --use-NN -R $path_atlas -i morphed_to_atlasAffine.txt morphed_to_atlasInverseWarp.nii.gz
# pet scan is transformed into atlas space
# WarpImageMultiTransform 3 $path_pet pet_to_atlas.nii -R $path_atlas -i morphed_to_atlasAffine.txt morphed_to_atlasInverseWarp.nii.gz
