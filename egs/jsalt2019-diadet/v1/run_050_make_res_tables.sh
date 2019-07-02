# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

diar_name=slid_win
net_name=2a.1.voxceleb_div2

. parse_options.sh || exit 1;

be_name=lda200_splday150_v1_voxceleb_div2
score_dir=exp/scores/$net_name/${be_name}

#Video table
args="--print_header true"
dirs=(plda plda_gtvad plda_${diar_name})
cases=("w/o diar" "ground-truth diar" "${diar_name}")

nc=${#dirs[*]}
for((i=0;i<$nc;i++))
do
    d=${dirs[$i]}
    dir_i=$score_dir/${d}_cal_v1
    local/make_table_line.sh $args "${net_name} ${cases[$i]}" $dir_i
    args=""
done

echo ""
