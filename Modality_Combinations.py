from itertools import combinations
import nibabel as nib
import numpy as  np 
import os 
from pathlib import Path

### script to get various combinations of pre processed MRI data ###


def combination_gen(lst, names):
    """get various different combinations of modalities"""
    ### useful during exploratory analysis###
    comb_dict = {}
    for r in range(1, len(lst) + 1):
        for comb in combinations(zip(lst, names), r):
            arrays, modes = zip(*comb)
            modes_str = "_".join(modes)
            comb_dict[modes_str] = arrays
    return comb_dict




path  = 'tumor_voets_NF2/data_N2_after_norm'
save_path = 'tumor_voets_NF2/data_combinations'


def mod_combinations(modalities: list[str],path: Path, save_path: Path = None, save: bool = False,all_combs: bool = False)-> np.array:

    """Output an array of specified/ all the modality combinations for each patient
    may need to make changes to the paths as appropriate"""
    
    if modalities == []:
        raise ValueError("Please specify the modalities you want to combine")
    
    if modalities == ['T1'] or modalities == ['T2'] or modalities == ['T1C'] or modalities == ['FLAIR']:
        raise ValueError("Please specify more than one modality")
    
    cases = os.listdir(path)
    

    for case in cases: 
        list_modality_arrays = []
        modality_names = []


        for modality in modalities:

         
            modality_pth = os.path.join(path, case, f"{case}_{modality}.nii.gz")
            
            #if modality is missing continue to next modality so get combination of what is there. 
            try:
                modality_nib = nib.load(modality_pth)
            except FileNotFoundError:
                continue 
                
                
            # get affine values from T1 for saving image later
            modality_affine = modality_nib.affine
            modality_array = modality_nib.get_fdata()
            modality_names.append(modality)   # dont need this will leave for now. 
            list_modality_arrays.append(modality_array)

        if all_combs:
            mode_combs = combination_gen(list_modality_arrays, modality_names)
        else:
            mode_combs = {f"{modalities[0]}_{modalities[1]}": list_modality_arrays}

        for mode_str, arrays in mode_combs.items():
                
            combined_img = np.stack(arrays, axis=-1)
            combined_image = nib.Nifti1Image(combined_img, affine=modality_affine)

            print(combined_image.shape)
            
            if save:
                nib.save(combined_image, os.path.join(path, f"{case}_{mode_str}.nii.gz"))

        print(f"Done with Creating Desired Modality Combinations {case}")

    return combined_image




if __name__ == "__main__":

    # gives me all combinations of specificied modalities, either all combinations or specific ones. 
     
    path  = 'tumor_voets_NF2/data_N2_after_norm'
    save_path = None
    mod_combinations(modalities = ['T1','T2','T1C'], path = path,all_combs = True)

    