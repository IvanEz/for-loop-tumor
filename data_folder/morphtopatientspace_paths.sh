#!/bin/sh

path_patient="t1.nii.gz"
path_atlas="atlas_resized_229to129.nii"
path_tumor_sim=$1
path_tumor_sim_patientspace=$2


if test -f "$path_patient"; then
    echo "$path_patient exists."
else
    path_patient="t1c.nii.gz"
fi

echo $path_tumor_sim_patientspace
echo $path_tumor_sim

#forward -> atlas is transformed into patient space
WarpImageMultiTransform 3 $path_tumor_sim $path_tumor_sim_patientspace -R $path_patient morphed_to_atlasWarp.nii.gz morphed_to_atlasAffine.txt

