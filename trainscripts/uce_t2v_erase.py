from collections import defaultdict
import torch
torch.set_grad_enabled(False)
import argparse
import os
import copy
import time

from safetensors.torch import save_file
from diffusers import TextToVideoSDPipeline, DPMSolverMultistepScheduler


def UCE(pipe, edit_concepts, guide_concepts, preserve_concepts, erase_scale, preserve_scale, lamb, save_dir, exp_name):
    start_time = time.time()
    
    # Prepare the cross attention weights required to do UCE
    uce_modules = dict()
    all_modules = dict(pipe.unet.named_modules())
    
    print(list(filter(lambda x: 'attn2' in x, all_modules.keys())))   
    
    for name, module in all_modules.items():
        # Exclude temporal models (transformer_in, temp_attentions) - only process spatial Transformer2DModel modules
        if 'attn2' in name and 'temp_attentions' not in name and 'transformer_in' not in name and (name.endswith('to_v') or name.endswith('to_k')):
            print(f"Selecting module for UCE: {name}")
            uce_modules[name] = module

            
    original_modules = copy.deepcopy(uce_modules)
    uce_modules = copy.deepcopy(uce_modules)

    # collect text embeddings for erase concept and retain concepts
    uce_erase_embeds = {}
    for e in edit_concepts + guide_concepts + preserve_concepts:
        if e in uce_erase_embeds:
            continue
        t_emb = pipe.encode_prompt(prompt=e,
                                   device=device,
                                   num_images_per_prompt=1,
                                   do_classifier_free_guidance=False)
    
        last_token_idx = (pipe.tokenizer(e,
                                          padding="max_length",
                                          max_length=pipe.tokenizer.model_max_length,
                                          truncation=True,
                                          return_tensors="pt",
                                         )['attention_mask']).sum()-2
    
        # Extract the last token embedding
        base_emb = t_emb[0][:,last_token_idx,:]
        
        print('Base embedding shape:', base_emb.shape)
    
        uce_erase_embeds[e] = base_emb
    
    # collect cross attention outputs for guide concepts and retain concepts (this is for original model weights)
    uce_guide_outputs = defaultdict(dict)
    for g in guide_concepts + preserve_concepts:
        if g in uce_guide_outputs:
            continue
            
        t_emb = uce_erase_embeds[g]
        
        for name, module in original_modules.items():
            uce_guide_outputs[g][name] = module(t_emb)

    ###### UCE Algorithm (variables are named according to the paper: https://arxiv.org/abs/2308.14761)
    for name, module in original_modules.items():
        # get original weight of the model
        w_old = module.weight

        # for the left hand term in equation 7 from the paper
        mat1 = lamb * w_old
        # for the right hand term in equation 7 from the paper (we will inverse this later)
        mat2 = lamb * torch.eye(w_old.shape[1], device = device, dtype=torch_dtype)  
    
        # Erase Concepts
        for erase_concept, guide_concept in zip(edit_concepts, guide_concepts):
            # Use the caption-projected embedding for this specific module
            c_i = uce_erase_embeds[erase_concept].T
            v_i_star = uce_guide_outputs[guide_concept][name].T
    
            mat1 += erase_scale * (v_i_star @ c_i.T)
            mat2 += erase_scale * (c_i @ c_i.T)
    
        # Retain Concepts
        for preserve_concept in preserve_concepts:
            # Use the caption-projected embedding for this specific module
            c_i = uce_erase_embeds[preserve_concept].T
            v_i_star = uce_guide_outputs[preserve_concept][name].T
    
            mat1 += preserve_scale * (v_i_star @ c_i.T)
            mat2 += preserve_scale * (c_i @ c_i.T)
    
    
        uce_modules[name].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2.float()).to(torch_dtype))
    
    # save the weights
    uce_state_dict = {}
    for name, parameter in uce_modules.items():
        uce_state_dict[name+'.weight'] = parameter.weight
    save_file(uce_state_dict, os.path.join(save_dir, exp_name+'.safetensors'))
    
    end_time = time.time()
    print(f'\n\nErased concepts using UCE\nModel edited in {end_time-start_time} seconds\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUCE_T2V',
                    description = 'UCE for erasing concepts in Text-to-Video Models')
    parser.add_argument('--edit_concepts', help='prompts corresponding to concepts to erase separated by ;', type=str, required=True)
    parser.add_argument('--guide_concepts', help='Concepts to guide the erased concepts towards seperated by ;', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve seperated by ;', type=str, default=None)
    parser.add_argument('--concept_type', help='type of concept being erased', choices=['art', 'object', 'character', 'action', 'unsafe'], type=str, required=True)
    
    parser.add_argument('--model_id', help='Model to run UCE on', type=str, default="cerspense/zeroscope_v2_576w")
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='cuda:0')
    
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=1)
    parser.add_argument('--lamb', help='lambda regularization term for UCE', type=float, required=False, default=0.5)
    
    parser.add_argument('--expand_prompts', help='do you wish to expand your prompts?', choices=['true', 'false'], type=str, required=False, default='false')
    
    parser.add_argument('--save_dir', help='where to save your uce model weights', type=str, default='../uce_models')
    parser.add_argument('--exp_name', help='Use this to name your saved filename', type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device
    torch_dtype = torch.float16
    model_id = args.model_id
    
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    lamb = args.lamb
    
    concept_type = args.concept_type
    expand_prompts = args.expand_prompts
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = 'uce_t2v_test'

    # erase concepts
    edit_concepts = [concept.strip() for concept in args.edit_concepts.split(';')]
    # guide concepts
    guide_concepts = args.guide_concepts 
    if guide_concepts is None:
        guide_concepts = ''
        if concept_type == 'art':
            guide_concepts = 'art'
        elif concept_type == 'character':
            guide_concepts = 'person'
        elif concept_type == 'action':
            guide_concepts = 'person doing something'
    guide_concepts = [concept.strip() for concept in guide_concepts.split(';')]
    if len(guide_concepts) == 1:
        guide_concepts = guide_concepts*len(edit_concepts)
    if len(guide_concepts) != len(edit_concepts):
        raise Exception('Error! The length of erase concepts and their corresponding guide concepts do not match. Please make sure they are seperated by ; and are of equal sizes')

    # preserve concepts
    if args.preserve_concepts is None:
        preserve_concepts = []
    else:
        preserve_concepts = [concept.strip() for concept in args.preserve_concepts.split(';')]
    
    

    if expand_prompts == 'true':
        edit_concepts_ = copy.deepcopy(edit_concepts)
        guide_concepts_ = copy.deepcopy(guide_concepts)

        for concept, guide_concept in zip(edit_concepts_, guide_concepts_):
            if concept_type == 'art':
                edit_concepts.extend([f'video of {concept} style',
                                       f'{concept} style animation',
                                       f'{concept} style video',
                                       f'footage in {concept} style',
                                       f'{concept} style scene'
                                      ]
                                     )
                guide_concepts.extend([f'video of {guide_concept} style',
                                       f'{guide_concept} style animation',
                                       f'{guide_concept} style video',
                                       f'footage in {guide_concept} style',
                                       f'{guide_concept} style scene'
                                      ]
                                     )
            elif concept_type in ['character', 'object']:
                edit_concepts.extend([f'video of {concept}',
                                       f'{concept} in video',
                                       f'footage of {concept}',
                                       f'{concept} on screen',
                                       f'scene with {concept}'
                                      ]
                                     )
                guide_concepts.extend([f'video of {guide_concept}',
                                       f'{guide_concept} in video',
                                       f'footage of {guide_concept}',
                                       f'{guide_concept} on screen',
                                       f'scene with {guide_concept}'
                                      ]
                                     )
            elif concept_type == 'action':
                edit_concepts.extend([f'person {concept}',
                                       f'someone {concept}',
                                       f'video of {concept}',
                                       f'footage of {concept}',
                                       f'{concept} action'
                                      ]
                                     )
                guide_concepts.extend([f'person {guide_concept}',
                                       f'someone {guide_concept}',
                                       f'video of {guide_concept}',
                                       f'footage of {guide_concept}',
                                       f'{guide_concept} action'
                                      ]
                                     )
            else:
                edit_concepts.extend([f'video of {concept}',
                                       f'footage of {concept}',
                                       f'scene of {concept}',
                                       f'{concept} in video',
                                       f'video showing {concept}'
                                      ]
                                     )
                guide_concepts.extend([f'video of {guide_concept}',
                                       f'footage of {guide_concept}',
                                       f'scene of {guide_concept}',
                                       f'{guide_concept} in video',
                                       f'video showing {guide_concept}'
                                      ]
                                     )


    print(f"\n\nErasing: {edit_concepts}\n")
    print(f"Guiding: {guide_concepts}\n")
    print(f"Preserving: {preserve_concepts}\n")
    
    pipe = TextToVideoSDPipeline.from_pretrained(model_id, 
                                                  torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    print(dict(pipe.unet.named_modules()).keys())
    
    UCE(pipe, edit_concepts, guide_concepts, preserve_concepts, erase_scale, preserve_scale, lamb, save_dir, exp_name)
