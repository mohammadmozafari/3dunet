EXP_NAME="3dunet-semi"
EXP_DIR="./exps/${EXP_NAME}"
mkdir -p $EXP_DIR

OUTPUT_FILE="${EXP_DIR}/$(date +"%Y-%m-%d-%T").txt"
python -u run.py | tee $OUTPUT_FILE