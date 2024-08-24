from glob import glob

output_dir = "tmp/output"

eval_output_paths = glob(output_dir + "/**/eval_outputs.json", recursive=True)

# orig_evals - has eval_outputs.json in parent output directory, indicates a job that
# finished successfully
model_dirs = [
    path.removesuffix("/eval_outputs.json")
    for path in eval_output_paths
    if "/eval_outputs/" not in path
]

# we have re-run some evals and saved the updated results to an eval_outputs
# sub-directory. Find the models that don't have this.
not_rerun_evals = [
    path
    for path in model_dirs
    if not any(path + "/eval_outputs/" in p for p in eval_output_paths)
]
print(len(model_dirs))
print(len(not_rerun_evals))

# the models that do have re-run eval outputs
rerun_evals = [path for path in model_dirs if path not in not_rerun_evals]
print(len(rerun_evals))

# some models we have also saved eval outputs on the training data. Find the models
# that don't have this, out of the ones that do have re-run normal eval outputs.
no_train_eval = [
    path
    for path in rerun_evals
    if len(glob(path + "/eval_outputs/**/train_set_eval_outputs.json")) == 0
]
print(len(no_train_eval))
