
import torch
from imagefilter3d import GINGroupConv3D
import adv_bias as AdvBias
from adv_bias import AdvBias3D, rescale_intensity





class config():
    # config for gin
    nb_gin = 20
    gin_out_nc = 3 # fit into the network
    gin_n_interm_ch = 2
    gin_nlayer = 4
    gin_norm = 'frob'

    # config for ipa correlation maps
    blend_grid_size = 24 # 24*2=48, 1/4 of image size
    blend_epsilon = 0.3





def set_networks(self, opt):

        # data augmentation nodes
        if opt.exp_type == 'gin':
            self.img_transform_node = GINGroupConv3D(out_channel = opt.gin_out_nc, n_layer = opt.gin_nlayer, interm_channel = opt.gin_n_interm_ch, out_norm = opt.gin_norm).cuda()
        elif opt.exp_type == 'ginipa':
            # ipa
            blender_cofig = {
                    'epsilon': opt.blend_epsilon,
                    'xi': 1e-6,
                    'control_point_spacing':[opt.blend_grid_size, opt.blend_grid_size],
                    'downscale':2, #
                    'data_size':[opt.batchSize,1,opt.fineSize, opt.fineSize],
                    'interpolation_order':2,
                    'init_mode':'gaussian',
                    'space':'log'
                    }

            self.img_transform_node = GINGroupConv3D(out_channel = opt.gin_out_nc, n_layer = opt.gin_nlayer, interm_channel = opt.gin_n_interm_ch, out_norm = opt.gin_norm).cuda()
            self.blender_node       = AdvBias(blender_cofig) # IPA
            self.blender_node.init_parameters()

        else:
            raise NotImplementedError(f'Unknown exp type: {opt.exp_type}')

        print(f'Using image transform type {opt.exp_type}')











def set_input_aug_sup(self, input):
    '''
    Applying both GIN and IPA

    '''
    input_img   = input['img']
    input_mask  = input['lb']

    if len(self.gpu_ids) > 0:
        input_img = input_img.float().cuda(self.gpu_ids[0])
        input_mask = input_mask.float().cuda(self.gpu_ids[0])

    # random no-linear augmentation
    self._nb_current = input_img.shape[0] # batch size of the current batch

    # gin
    input_buffer = torch.cat([  self.img_transform_node(input_img) for ii in range(3)], dim = 0)

    if 'ipa' in self.opt.exp_type:

        self.blender_node.init_parameters()
        blend_mask = rescale_intensity(self.blender_node.bias_field).repeat(1,3,1,1)

        # spatially-variable blending
        input_cp1 = input_buffer[: self._nb_current].clone().detach() * blend_mask + input_buffer[self._nb_current: self._nb_current * 2].clone().detach() * (1.0 - blend_mask)
        input_cp2 = input_buffer[: self._nb_current] * (1 - blend_mask) + input_buffer[self._nb_current: self._nb_current * 2] *  blend_mask

        input_buffer[: self._nb_current] = input_cp1
        input_buffer[self._nb_current: self._nb_current * 2] = input_cp2

        self.blend_mask = blend_mask.data

    self.input_img_3copy = input_buffer
    self.input_mask = input_mask




if __name__ == "__main__":
    # Add your main logic here
    pass