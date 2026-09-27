"""Evaluate the released V-JEPA P-only, post-hoc F and goal-G/F controls."""
import argparse
import json
from pathlib import Path
import sys


def validate_export_scope(cfg):
    if not isinstance(cfg, dict):
        raise ValueError('Evaluation configuration must be a mapping.')
    deployment = cfg.get('deployment')
    if deployment not in {'proposal_only', 'posthoc_f', 'goal_gf'}:
        raise ValueError('Choose proposal_only, posthoc_f or goal_gf; use the separate Full CLI for Full.')
    visual = cfg.get('visual', {})
    if not isinstance(visual, dict) or visual.get('encoder_family') is not None:
        raise ValueError('This entrypoint provides the default V-JEPA frontend only.')
    interval = cfg.get('diagnostic_execution_interval', 1)
    if type(interval) is not int or interval != 1:
        raise ValueError('The released evaluation executes one control step per plan.')
    if 'diagnostic_lwm_rerank' in cfg or 'diagnostic_stagnation_guard' in cfg:
        raise ValueError('Additional scoring or recovery adapters are outside these controls.')
    options = cfg.get('diagnostic_whole_branch_ranking')
    if deployment == 'proposal_only':
        if 'diagnostic_whole_branch_ranking' in cfg:
            raise ValueError('P-only ranks Q endpoints without F whole-branch scoring.')
    else:
        if not isinstance(options, dict):
            raise ValueError('Explicit V-JEPA ranking options are required.')
        if options.get('score_mode', 'l1_consistency') != 'l1_consistency':
            raise ValueError('Use the released V-JEPA L1 score.')
        if deployment == 'goal_gf' and options.get('risk_weight') != 0:
            raise ValueError('The no-Q control requires endpoint-only risk_weight=0.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    validate_export_scope(cfg)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from j2j_recurrent_experiments.closed_loop.runner import run_closed_loop_configuration
    run_closed_loop_configuration(config=cfg, output_path=args.output)


if __name__ == '__main__':
    main()
