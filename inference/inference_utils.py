import os
import sys
import tqdm
import re
import torch
from collections import defaultdict
from os.path import dirname, abspath, join

file_path = abspath(dirname(__file__))
sys.path.append(abspath(join(file_path, os.pardir, 'model')))
sys.path.append(abspath(join(file_path, os.pardir, 'model', 'nlgeval')))

from eval_utils.decode import greedy_with_att
from engine import pack_encoder_inputs, normalize_with_tokenizer
from nlgeval import NLGEval
from eval_model import prepare_model, setup_val_dataloader
from eval_utils.decode import prepare_tokenizer


def override_config_with_checkpoint(checkpoint, config):
    # parse checkpoint filename
    pattern = '([a-z]+)_([a-z\+]+)_([\w_]+)_checkpoint_(\d)+.pth'
    architecture, dataset, info, epoch = re.findall(pattern, checkpoint)[0]  

    use_glob = config.use_global_features
    use_loc = config.use_location_features
    prefix = config.prefix

    if dataset != prefix:
        config.prefix = dataset
        print(f'''CAUTION: Overriding configuration!
                WAS: prefix=={prefix};
                NEW: prefix=={config.prefix}''')
        config.ref_dir = join(config.ref_base, config.prefix)
    if 'loc' in info:
        if not use_loc:
            config.use_location_features = True
            print(f'''CAUTION: Overriding configuration!
                WAS: use_location_features=={use_loc};
                NEW: use_location_features=={config.use_location_features}''')
    if 'glob' in info:
        if not use_glob:
            config.use_global_features = True
            print(f'''CAUTION: Overriding configuration!
                WAS: use_global_features=={use_glob};
                NEW: use_global_features=={config.use_global_features}''')


def eval_model_with_att(model, data_loader, tokenizer,
               config, metrics_to_omit=[],
               print_samples=False):
    """
    iterate through val_loader and calculate CIDEr scores for model
    (only works with batch_size=1 for now)
    """

    model.eval()

    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True,
        metrics_to_omit=metrics_to_omit
    )

    # construct reference dict
    annotations = defaultdict(list)
    for a in data_loader.dataset.annot:
        annotations[a[0]].append(a[2])

    hypotheses, ids_hypotheses, references = [], [], []

    bos_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    global_features = data_loader.dataset.return_global_context
    location_features = data_loader.dataset.return_location_features

    # decode imgs in val set
    for i, (ann_ids, *encoder_input, _, _) in enumerate(tqdm.tqdm(data_loader)):

        samples = pack_encoder_inputs(encoder_input, global_features, location_features)

        # get model predictions (with greedy/att function)
        hyp_ids, atts = greedy_with_att(
            model, samples, tokenizer, 
            max_pos_embeddings=config.max_position_embeddings, 
            start_token=bos_id, 
            end_token=eos_id, 
            return_raw=True
        )

        decoded_hyp = tokenizer.decode(hyp_ids, skip_special_tokens=True)

        hypotheses.append(decoded_hyp)

        # PROCESS ATTENTIONS

        encoder_atts = [a['enc_tc_self_att'] for a in atts]
        decoder_atts = [a['dec_exp_tc_cross_att'] for a in atts]
        stacked_encoder_atts = torch.stack(encoder_atts) # [token, layer, batch, features, vis_parts]
        stacked_decoder_atts = torch.stack(decoder_atts) # [token, layer, batch, features, vis_parts]

        # reduce

        stacked_encoder_atts = stacked_encoder_atts[:, :, 0, :, :] # [token, layer, features, vis_parts]
        stacked_decoder_atts = stacked_decoder_atts[:, :, 0, :, :] # [token, layer, features, vis_parts]

        stacked_encoder_atts = stacked_encoder_atts.mean(-2)  # [token, layer, vis_parts]
        stacked_decoder_atts = stacked_decoder_atts.mean(-2)  # [token, layer, vis_parts]


        ids_hypotheses.append(
            {
                'ann_id': ann_ids.item(),
                'expression_ids': hyp_ids.tolist(),
                'encoder_attentions': stacked_encoder_atts.detach().cpu().numpy(),
                'decoder_attentions': stacked_decoder_atts.detach().cpu().numpy(),
                'expression_string': decoded_hyp
            }
        )

        if print_samples:
            print(decoded_hyp)

        # get annotated references
        refs = [annotations[i.item()] for i in ann_ids]
        normalized_refs = [
            [normalize_with_tokenizer(r, tokenizer) for r in _refs] for _refs in refs
        ]
        references += normalized_refs

    # transpose references to get correct format
    transposed_references = list(map(list, zip(*references)))

    # calculate cider score from hypotheses and references
    metrics_dict = nlgeval.compute_metrics(
        ref_list=transposed_references, hyp_list=hypotheses)

    return metrics_dict, ids_hypotheses


def main_val_set_with_att(args, config):

    # model
    model = prepare_model(args, config).to(args.device)
    print(f'Successfully loaded {model.__class__.__name__} model')

    # tokenizer
    tokenizer, _, _ = prepare_tokenizer()

    data_loader = setup_val_dataloader(config, args.split)

    metrics, generated = eval_model_with_att(
        model, data_loader, tokenizer, config, print_samples=args.print_samples
    )

    return metrics, generated