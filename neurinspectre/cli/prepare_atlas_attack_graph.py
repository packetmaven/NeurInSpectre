"""
Prepare a real AI attack case summary mapped to MITRE ATLAS phases.

This script builds a minimal nodes/edges JSON from well-documented attack
families observed in the literature (e.g., jailbreak/prompt injection,
model extraction, data poisoning/backdoor). It is not a dataset dump; rather
it encodes a concise scenario that the ATLAS visualizer can render.
"""

import argparse
import json
from pathlib import Path


SCENARIOS = {
    'jailbreak_extraction': {
        'nodes': [
            {'id': 'Execution', 'label': 'Execution', 'atlas_phase': 'Execution', 'atlas_id': 'AML.TA0005'},
            {'id': 'Defense Evasion', 'label': 'Defense Evasion', 'atlas_phase': 'Defense Evasion', 'atlas_id': 'AML.TA0007'},
            {'id': 'Collection', 'label': 'Collection', 'atlas_phase': 'Collection', 'atlas_id': 'AML.TA0009'},
            {'id': 'Exfiltration', 'label': 'Exfiltration', 'atlas_phase': 'Exfiltration', 'atlas_id': 'AML.TA0010'},
            # Techniques
            {'id': 'prompt_injection', 'label': 'Prompt Injection (adv templates)', 'atlas_phase': 'Execution', 'atlas_id': 'AML.T0051', 'metrics': {'success_rate': 0.62}},
            {'id': 'jailbreak', 'label': 'Jailbreak (role-play)', 'atlas_phase': 'Defense Evasion', 'atlas_id': 'AML.T0054', 'metrics': {'success_rate': 0.48}},
            {'id': 'model_extraction', 'label': 'Model Extraction (functional I/O)', 'atlas_phase': 'Exfiltration', 'atlas_id': 'AML.T0024.002', 'metrics': {'queries': 1.2e5}},
            {'id': 'tool_abuse', 'label': 'Tool Abuse (file/system calls)', 'atlas_phase': 'Execution', 'atlas_id': 'AML.T0053', 'metrics': {'calls': 340}},
        ],
        'edges': [
            {'source': 'Execution', 'target': 'prompt_injection'},
            {'source': 'prompt_injection', 'target': 'jailbreak'},
            {'source': 'jailbreak', 'target': 'Collection'},
            {'source': 'Collection', 'target': 'model_extraction'},
            {'source': 'Defense Evasion', 'target': 'Exfiltration'},
            {'source': 'model_extraction', 'target': 'Exfiltration'}
        ]
    },
    'poison_backdoor': {
        'nodes': [
            {'id': 'Persistence', 'label': 'Persistence', 'atlas_phase': 'Persistence', 'atlas_id': 'AML.TA0006'},
            {'id': 'Defense Evasion', 'label': 'Defense Evasion', 'atlas_phase': 'Defense Evasion', 'atlas_id': 'AML.TA0007'},
            {'id': 'Impact', 'label': 'Impact', 'atlas_phase': 'Impact', 'atlas_id': 'AML.TA0011'},
            {'id': 'data_poisoning', 'label': 'Data Poisoning (training set)', 'atlas_phase': 'Persistence', 'atlas_id': 'AML.T0020', 'metrics': {'poison_frac': 0.02}},
            {'id': 'backdoor', 'label': 'Backdoor (triggered neurons)', 'atlas_phase': 'AI Attack Staging', 'atlas_id': 'AML.T0043.004', 'metrics': {'ASR': 0.91}},
            {'id': 'watermark_removal', 'label': 'Watermark Removal', 'atlas_phase': 'Impact', 'atlas_id': 'AML.T0031', 'metrics': {'distortion': 0.03}},
        ],
        'edges': [
            {'source': 'Persistence', 'target': 'data_poisoning'},
            {'source': 'data_poisoning', 'target': 'backdoor'},
            {'source': 'backdoor', 'target': 'Impact'},
            {'source': 'Defense Evasion', 'target': 'watermark_removal'}
        ]
    }
}


def main():
    ap = argparse.ArgumentParser(description='Prepare ATLAS attack graph input JSON')
    ap.add_argument('--output', required=True, help='Output JSON path')
    ap.add_argument('--scenario', choices=list(SCENARIOS.keys()), default='jailbreak_extraction')
    args = ap.parse_args()

    data = SCENARIOS[args.scenario]
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data, indent=2))
    print(outp.as_posix())


if __name__ == '__main__':
    main()


