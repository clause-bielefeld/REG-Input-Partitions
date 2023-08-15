import sys
import pickle
import os
import os.path as osp
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import argparse
from itertools import count

from refxp_processor import RefExpProcessor

file_path = osp.abspath(osp.dirname(__file__))

sys.path.append(osp.join(file_path, os.pardir, 'model'))
from configuration import Config
from data_utils.utils import get_refcoco_df
from eval_utils.decode import prepare_tokenizer

sys.path.append(osp.join(file_path, os.pardir))
from inference.inference_utils import override_config_with_checkpoint

config = Config()

def mean_layer_att(att):
    """take the mean of several attention layers (layer dimension: 0)"""
    att = att.mean(0)
    return att


def last_n_layer_att(att, n=1):
    """take the mean of the last n attention layers"""
    if type(n) == int:
        att = att[-n:, :]
    att = mean_layer_att(att)
    return att


def decompose_att_vector(att, vis_dim=196):
    """split an attention vector in target/location/context (on first dimension)"""
    t_att = att[:vis_dim]
    l_att = att[vis_dim:-vis_dim]
    c_att = att[-vis_dim:]
    return t_att, l_att, c_att


def convert_bbox_format(bb):
    """convert bounding box from xywh to corner coordinates format"""
    # get bounding box coordinates (round since integers are required)
    x, y, w, h = round(bb[0]), round(bb[1]), round(bb[2]), round(bb[3])

    # calculate minimum and maximum values for x and y dimension
    x_min, x_max = x, x + w
    y_min, y_max = y, y + h

    return x_min, x_max, y_min, y_max

def get_target_context_delta(target_attention, context_attention, normalize=True):
    """get difference between target and context attention mass

    Args:
        target_attention (np.array or torch.tensor): attention mask for target portion
        context_attention (np.array or torch.tensor): attention mask for context portion
        normalize (bool, optional): Normalize to range [-1, 1]. Defaults to True.

    Returns:
        delta (float): absolute difference between attention mass on target and context
    """

    t_sum = target_attention.sum()
    c_sum = context_attention.sum()

    if normalize:
        # normalize so that t_sum + c_sum == 1 
        total_attention = t_sum + c_sum
        t_sum = t_sum / total_attention
        c_sum = c_sum / total_attention
    
    delta = t_sum - c_sum

    return delta


def aggregate_continuations(generated_sample, tokenizer):
    """
    aggregate the attentions for subword tokens
    (attention for words <- mean attentions for subword tokens)
    """

    counter = count()

    all_eids, all_eatts, all_datts = [], [], []
    # iterate through zipped expression ids, encoder attentions and decoder attentions
    for eid, eatt, datt in zip(
                generated_sample['expression_ids'], 
                generated_sample['encoder_attentions'], 
                generated_sample['decoder_attentions']
            ):

        decoded_token = tokenizer.decode(eid)
        
        if decoded_token.startswith('# #'):
            # for continuations: append to last word initial token lists
            all_eids[curr_idx].append(eid)
            all_eatts[curr_idx].append(eatt)
            all_datts[curr_idx].append(datt)
        else:
            # for word initial tokens: store in new lists
            curr_idx = next(counter)
            this_eid, this_eatt, this_datt = [eid], [eatt], [datt]
            all_eids.append(this_eid)
            all_eatts.append(this_eatt)
            all_datts.append(this_datt)

    # perform elementwise mean over aggregated attention maps
    elementwise_mean = lambda x: np.mean(x, axis=0)
    aggregated_eatts = list(map(elementwise_mean, all_eatts))
    aggregated_datts = list(map(elementwise_mean, all_datts))

    # stack attention maps to correspond to original shape, make output dict
    output_sample_dict = {
        'ann_id': generated_sample['ann_id'], 
        'expression_ids': all_eids, 
        'encoder_attentions': np.stack(aggregated_eatts, 0), 
        'decoder_attentions': np.stack(aggregated_datts, 0), 
        'expression_string': generated_sample['expression_string'],
    }

    return output_sample_dict


def att_sums(t_att, l_att, c_att, normalize_by_feature_dim=False, normalize_to_one=True):
    """return (normalized) sum of the attention weights on target, location and context"""
    
    t_att_sum = t_att.sum()
    l_att_sum = l_att.sum()
    c_att_sum = c_att.sum()

    if normalize_by_feature_dim:
        t_att_sum = t_att_sum / t_att.size
        l_att_sum = l_att_sum / l_att.size
        c_att_sum = c_att_sum / c_att.size

    if normalize_to_one:
        total_att_sum = t_att_sum + l_att_sum + c_att_sum
        t_att_sum = t_att_sum / total_att_sum
        l_att_sum = l_att_sum / total_att_sum
        c_att_sum = c_att_sum / total_att_sum
    
    return t_att_sum, l_att_sum, c_att_sum


def main(args):

    # init RefExpProcessor (for parsing and determining heads)
    processor = RefExpProcessor('en_core_web_trf')

    # file path stuff
    data_dir = args.data_dir
    path_base = osp.split(args.file_base)[-1]

    generated_file = path_base + 'generated.pkl'
    generated_path = os.path.join(data_dir, generated_file)

    if args.out_path is None:
        out_dir = osp.abspath(osp.join(data_dir, 'analysis_outputs'))
        if not osp.isdir(out_dir):
            print(f'create output directory {out_dir}')
            os.makedirs(out_dir)
        out_path = osp.abspath(osp.join(out_dir, path_base + f'analysis_last{args.last_n_layers}.json'))
    else:
        out_path = osp.abspath(args.out_path)

    print(f"""
        reading results from {generated_file}
        extracting attention from last {args.last_n_layers} layers
        saving results in {out_path}
    """.strip())

    if args.override_config:
        # adapt config to specifications of input file
        model_file_path = re.sub(r'_(val|testa|testb|test)_', '', path_base) + '.pth'
        override_config_with_checkpoint(model_file_path, config)

    # load results
    with open(generated_path, 'rb') as f:
        generated_data = pickle.load(f)

    # prepare refcoco df
    refcoco_df = get_refcoco_df(config.ref_dir)
    refcoco_df = refcoco_df.reset_index().groupby('ann_id').agg({
        'sent_id': list, 'caption': list, 
        'ref_id': 'first', 'refcoco_split': 'first', 
        'coco_split': 'first', 'image_id': 'first', 
        'bbox': 'first', 'category_id':'first',
    }).reset_index()
    refcoco_df.index = refcoco_df.ann_id.values

    # prepare tokenizer
    tokenizer, _, _ = prepare_tokenizer()

    ###########
    # ANALYZE #
    ###########

    # initialize dataframe for results
    results_df = pd.DataFrame(columns=[
        'ann_id', 'expression', 'token',
        'pos', 'tag', 'is_head', 'is_stop',
        'e_t_att_sum', 'e_loc_att_sum', 'e_c_att_sum',
        'e_t_att_sum_dimnorm', 'e_loc_att_sum_dimnorm', 'e_c_att_sum_dimnorm',
        'e_tc_delta', 
        'd_t_att_sum', 'd_loc_att_sum', 'd_c_att_sum',
        'd_t_att_sum_dimnorm', 'd_loc_att_sum_dimnorm', 'd_c_att_sum_dimnorm',        
        'd_tc_delta', 
        ]).astype({'is_head':bool,'is_stop':bool})

    for g in tqdm(generated_data):  # iterate through generated samples

        agg_g = aggregate_continuations(g, tokenizer)

        ann_id = agg_g['ann_id']

        head, parsed = processor.parse_and_extract_head(agg_g['expression_string'])
        head_ids = {h['id'] for h in head}    

        if len(parsed) < config.max_position_embeddings - 1:
            # don't do this if max seq len is reached
            parsed.append({  # to account for final [SEP] token
                'id': len(parsed),
                'start': None,
                'end': None,
                'tag': 'EOS',
                'pos': 'META',
                'morph': '',
                'lemma': '[SEP]',
                'dep': None,
                'head': None
            })

        assert len(agg_g['expression_ids']) == len(parsed), 'expression ids are not aligned with spacy parsing!'

        for i, (generated_id, encoder_attention, decoder_attention, parsed_token) in enumerate(zip(
                agg_g['expression_ids'], agg_g['encoder_attentions'], agg_g['decoder_attentions'], parsed
            )):  # iterate through individual ids (mapped to encoder/decoder attentions and spacy parsing)

            generated_token = tokenizer.decode(generated_id)
            pos = parsed_token['pos']
            tag = parsed_token['tag']

            #####################
            # ENCODER ATTENTION #
            #####################

            encoder_att = last_n_layer_att(encoder_attention, args.last_n_layers)
            e_t_att, e_l_att, e_c_att = decompose_att_vector(encoder_att)
            # ratio between target / context
            e_att_delta = get_target_context_delta(e_t_att, e_c_att, normalize=True)
            # summed (and normalized) attention weights on partitions
            e_t_att_sum, e_l_att_sum, e_c_att_sum = att_sums(e_t_att, e_l_att, e_c_att, normalize_by_feature_dim=False, normalize_to_one=True)
            e_t_att_sum_dimnorm, e_l_att_sum_dimnorm, e_c_att_sum_dimnorm = att_sums(e_t_att, e_l_att, e_c_att, normalize_by_feature_dim=True, normalize_to_one=True)

            #####################
            # DECODER ATTENTION #
            #####################

            decoder_att = last_n_layer_att(decoder_attention, args.last_n_layers)
            d_t_att, d_l_att, d_c_att = decompose_att_vector(decoder_att)
            # ratio between target / context
            d_att_delta = get_target_context_delta(d_t_att, d_c_att, normalize=True)
            # summed (and normalized) attention weights on partitions
            d_t_att_sum, d_l_att_sum, d_c_att_sum = att_sums(d_t_att, d_l_att, d_c_att, normalize_by_feature_dim=False, normalize_to_one=True)
            d_t_att_sum_dimnorm, d_l_att_sum_dimnorm, d_c_att_sum_dimnorm = att_sums(d_t_att, d_l_att, d_c_att, normalize_by_feature_dim=True, normalize_to_one=True)

            ################
            # SAVE RESULTS #
            ################

            results_df = pd.concat([
                results_df, 
                pd.DataFrame([{
                        # general info
                        'ann_id': ann_id, 'expression': agg_g['expression_string'], 'token': generated_token, 
                        'pos': pos, 'tag': tag, 'is_head': parsed_token['id'] in head_ids, 'is_stop': parsed_token['lemma'].lower() in processor.stopwords,
                        'token_idx': i,
                        # encoder results
                        'e_tc_delta': e_att_delta, 
                        'e_t_att_sum': e_t_att_sum, 'e_loc_att_sum': e_l_att_sum, 'e_c_att_sum': e_c_att_sum,
                        'e_t_att_sum_dimnorm': e_t_att_sum_dimnorm, 'e_loc_att_sum_dimnorm': e_l_att_sum_dimnorm, 'e_c_att_sum_dimnorm': e_c_att_sum_dimnorm,
                        # decoder results
                        'd_tc_delta': d_att_delta, 
                        'd_t_att_sum': d_t_att_sum, 'd_loc_att_sum': d_l_att_sum, 'd_c_att_sum': d_c_att_sum,
                        'd_t_att_sum_dimnorm': d_t_att_sum_dimnorm, 'd_loc_att_sum_dimnorm': d_l_att_sum_dimnorm, 'd_c_att_sum_dimnorm': d_c_att_sum_dimnorm,
                }])
            ], ignore_index=True, axis=0)

    # save to file
    results_df.to_json(out_path, orient='records')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('file_base')
    parser.add_argument('--last_n_layers', type=int)
    parser.add_argument('--data_dir', default=os.path.join(file_path, os.pardir, 'data', 'results'))
    parser.add_argument('--out_path', default=None, type=str)
    parser.add_argument('--override_config', action='store_true')

    args = parser.parse_args()

    if not args.last_n_layers:
        args.last_n_layers = 'all'

    main(args)