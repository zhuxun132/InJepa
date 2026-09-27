"""CLI for the released InJepa V-JEPA Full evaluation configuration."""
import argparse
import json
from pathlib import Path
import sys

def validate_export_scope(cfg):
    if not isinstance(cfg,dict):
        raise ValueError('Evaluation configuration must be a mapping.')
    if cfg.get('deployment')!='f_feedback_q':
        raise ValueError('This entrypoint evaluates InJepa Full (F to Q and G).')
    visual=cfg.get('visual',{})
    if visual.get('encoder_family') is not None:
        raise ValueError('This release provides the default V-JEPA frontend only.')
    options=cfg.get('diagnostic_whole_branch_ranking')
    if not isinstance(options,dict):
        raise ValueError('Explicit V-JEPA ranking options are required.')
    if options.get('score_mode','l1_consistency')!='l1_consistency':
        raise ValueError('Use the released V-JEPA L1 consistency score.')
    if cfg.get('diagnostic_execution_interval',1)!=1:
        raise ValueError('The released evaluation executes one control step per plan.')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    cfg=json.loads(args.config.read_text())
    validate_export_scope(cfg)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from j2j_recurrent_experiments.closed_loop.runner import run_closed_loop_configuration
    run_closed_loop_configuration(config=cfg,output_path=args.output)

if __name__=='__main__':main()
