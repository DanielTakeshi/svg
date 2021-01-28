# Perform model predictions.

# ---------------------------------------- #
# FabricV1
# ---------------------------------------- #

#for MODEL in model_0050 model_0100 model_0200 model_0400 model_0600 model_0800 model_1000 model_1200 model_1499; do
#    python compare_sv2p_svg.py  --datatype=fabric-random  --svg_model=${MODEL}
#done

for MODEL in model_0050 model_0100 model_0200 model_0400 ; do
    python compare_sv2p_svg.py  --datatype=fabric-random  --svg_model=${MODEL}
done


# ---------------------------------------- #
# FabricV2
# ---------------------------------------- #

#for MODEL in model_0050 model_0100 model_0200 model_0400 model_0600 model_0800 model_1000 model_1200 model_1499 ; do
#    python compare_sv2p_svg.py  --datatype=fabric-01-2021  --svg_model=${MODEL}
#done

for MODEL in model_0050 model_0100 model_0200 model_0400 model_0600 ; do
    python compare_sv2p_svg.py  --datatype=fabric-01-2021  --svg_model=${MODEL}
done
