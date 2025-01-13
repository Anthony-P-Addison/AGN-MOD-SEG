from old.test import create_dataset_for_test, test
from old.Modality_Combinations import combination_gen,mod_combinations
import argparse
import torch
import utils
import config
import os


### create combninations of modalities and test them with pretrained model and return dice scores for each ##

path = 'data_N2_after_norm'
labels= os.path.join(path, 'labels')
images = os.path.join(path, 'images')




if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument("--datasets_to_test", help="dataset for testing", type=str)
    parser.add_argument("--modalities_to_test", help="The modalities for testing (the index of the modalities for that input),using '_' to separate if 0_1_2 for BRATS it would mean test on FLAIR, T1, T1c", type=str)
    parser.add_argument("--test_all_combinations", help="0 or 1 1 if testing on all possible modality combinations", type=int, default='0')
    args = parser.parse_args()


    # want to create desired combinations and then test them with pretrained model. 

    path = 'data_N2_after_norm'
    labels= os.path.join(path, 'labels')
    images = os.path.join(path, 'images')


    for combinations in ['0']:
      for dataset in ['VOETS2']:
        for modality in ["0_1"]:

            args = argparse.Namespace()
            args.test_all_combinaions = combinations 
            args.datasets_to_test = dataset
            args.modalities_to_test = modality
            test_all_combinations=bool(args.test_all_combinations)
            cuda_id = "cuda:" + str(0)
            device = torch.device(cuda_id)
            torch.cuda.set_device(cuda_id)
            results = {}    
            Test_config=config.Test_config()

            ####

            Test_config.model_file_path = "models/Train_BRATS_TBI_ATLAS_MSSEG_WMH.pth"

            #####
        

            print("*************** TESTING NET " + str(Test_config.model_file_path) + " **************")        

            model = utils.create_net(Test_config.model_file_path,Test_config.model_net_type,Test_config.model_modalities_trained_on, device, cuda_id)

            print("************** TESTING DATASET " + args.datasets_to_test + " ***************")
            dataloader = create_dataset_for_test(args.datasets_to_test)
            if test_all_combinations:
                        modalities = utils.create_modality_combinations([int(x) for x in args.modalities_to_test.split("_")])
            else:
                        modalities = [[int(x) for x in args.modalities_to_test.split("_")]]

            for combination in modalities:
                        print(combination)
                        dsc_scores, seg_pix_vols, gt_pix_vols = test(model,
                            dataloader,
                            args.datasets_to_test,
                            combination,
                            Test_config.model_net_type,
                            Test_config.model_modalities_trained_on,
                            Test_config.model_channel_map,
                            device,
                            save_outputs= Test_config.save_segs,
                            save_path=Test_config.save_path,
                            )