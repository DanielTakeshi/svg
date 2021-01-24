# Perform model predictions.
HEAD=/data/svg/logs-Jan-24-SVG-1500epochs

# ---------------------------------------- #
# dold (old data), 80-20 train-valid split #
# ---------------------------------------- #

for MODEL in model_0050 model_0100 model_0200 model_0400 model_0600 model_0800 model_1000 model_1200 model_1499 ; do
    python compare_sv2p_svg.py  --datatype=fabric-random  --svg_model=${MODEL}
done

# ---------------------------------------- #
# dnew (new data), 80-20 train-valid split #
# ---------------------------------------- #

for MODEL in model_0050 model_0100 model_0200 model_0400 model_0600 model_0800 model_1000 model_1200 model_1499 ; do
    python compare_sv2p_svg.py  --datatype=fabric-01-2021  --svg_model=${MODEL}
done
