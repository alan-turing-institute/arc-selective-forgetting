"""
Add additional metrics to old eval output files.
"""

from glob import glob

from tqdm import tqdm

from arcsf.eval.evaluate import EvaluateOutputs

output_dir = "output"

eval_output_files = glob(output_dir + "/**/*eval_outputs*.json", recursive=True) + glob(
    output_dir + "/**/eval_checkpoints/*.json", recursive=True
)

print(f"Updating {len(eval_output_files)} output files with new metrics...")
for eval_output_file in tqdm(eval_output_files):
    eval_outputs = EvaluateOutputs.load(eval_output_file)
    eval_outputs._add_additional_metrics()
    eval_outputs.save(eval_output_file)
