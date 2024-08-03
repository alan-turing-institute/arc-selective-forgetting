EXPDIR=$(dirname $0)/$1
FULLID=$(sbatch --parsable $EXPDIR/full.sh)
echo "FULLID: $FULLID"
RETAINID=$(sbatch --parsable $EXPDIR/retain.sh)
echo "RETAINID: $RETAINID"
FORGETID=$(sbatch --parsable --dependency=afterok:$FULLID:$RETAINID --kill-on-invalid-dep=yes $EXPDIR/forget.sh)
echo "FORGETID: $FORGETID"
