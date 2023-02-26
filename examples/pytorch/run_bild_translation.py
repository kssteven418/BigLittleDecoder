import os
import subprocess
import argparse
from pprint import pprint

def arg_parse():
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--large', type=str, default=None)
    parser.add_argument('--small', type=str, default=None)
    parser.add_argument('--initialize_decoder_small_with', type=str, default=None) # model name or path to fix decoder-small with
    parser.add_argument('--dataset_name', type=str, required=True, choices=['wmt14', 'iwslt2017'])
    parser.add_argument('--dataset_config', type=str, required=True)
    parser.add_argument('--source_lang', type=str, required=True)
    parser.add_argument('--target_lang', type=str, required=True)

    parser.add_argument('--bild_fallback_threshold', type=float, default=None)
    parser.add_argument('--bild_rollback_threshold', type=float, default=None)

    args = parser.parse_args()
    return args

args = arg_parse()

if 'en-de' in args.model:
    assert args.target_lang == 'de' and args.source_lang == 'en'
if 'de-en' in args.model:
    assert args.target_lang == 'en' and args.source_lang == 'de'


FILE = 'translation/run_translation.py'

subprocess_args = [
    'python', FILE,
    # evaluate flags
    '--metric_for_best_model', 'bleu', 
    '--greater_is_better', 'True',
    '--predict_with_generate',
    '--num_beam', '1',

    # save flags
    '--save_total_limit', str(5),
    '--load_best_model_at_end',
    '--logging_steps', str(500),
    '--output_dir', './temp',

    # dataset flags
    '--dataset_name', args.dataset_name,
    '--dataset_config', args.dataset_config,
    '--source_lang', args.source_lang,
    '--target_lang', args.target_lang,
    '--model_name_or_path', args.model,
    '--evaluation_strategy', 'epoch',
    '--save_strategy', 'epoch',
    '--do_eval',
    '--per_device_eval_batch_size', '1',
]

# Avoid wandb logging, remove it to enable it
os.environ["WANDB_DISABLED"] = "true"

if args.bild_fallback_threshold:
    subprocess_args += ['--fallback_threshold', str(args.bild_fallback_threshold)]

if args.bild_rollback_threshold:
    subprocess_args += ['--rollback_threshold', str(args.bild_rollback_threshold)]

if args.model == 'bild':
    assert args.large is not None
    assert args.small is not None
    subprocess_args += [
        '--large', args.large, '--small', args.small
    ]

pprint(subprocess_args)

subprocess.call(subprocess_args)
