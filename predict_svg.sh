# ---------------------------------------------------------------------------------------------------------- #
# Perform model predictions. Load the network, and predict on held out data that I made in cloth-visual-mpc.
# Update: let's now use the better prediction w/better loading. Note `model_dir` instead of `model_path`.
# `results_svg_1/`: no learned action embedding, saving using PyTorch bad way
# `results_svg_2/`: no learned action embedding, saving using RECOMMENDED PyTorch way
# `results_svg_3/`: LEARNED action embedding, saving using RECOMMENDED PyTorch way
# ---------------------------------------------------------------------------------------------------------- #

# results_svg_1
#HEAD=/data/svg/logs-Jan-24-SVG-1500epochs

# results_svg_2
#HEAD=/data/svg/logs-Jan27-fabric-v1-v2-models-1500-epochs

# results_svg_2
HEAD=/data/svg/logs-Jan28-fabric-v1-v2-models-partially

# ---------------------------------------- #
# FabricV1, 80-20 train-valid split
# ---------------------------------------- #

# results_svg_1

#for MODEL in model_0050.pth model_0100.pth model_0200.pth model_0400.pth model_0600.pth model_0800.pth model_1000.pth model_1200.pth model_1499.pth ; do
#    python predict_svg_lp.py \
#        --model_path=${HEAD}/fabric-random/model=dcgan56x56-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=3-n_future=7-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000-act-cond-1/${MODEL}  \
#        --data_path=/home/seita/cloth-visual-mpc/logs/demos-fabric-random-epis_400_COMBINED.pkl
#done

# results_svg_2

#for MODEL in model_0050.pth model_0100.pth model_0200.pth model_0400.pth model_0600.pth model_0800.pth model_1000.pth model_1200.pth model_1499.pth ; do
#    python test_load_ssim.py \
#        --model_dir=${HEAD}/fabric-random/model=dcgan56x56-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=3-n_future=7-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000-act-cond-1/${MODEL}  \
#        --data_path=/home/seita/cloth-visual-mpc/logs/demos-fabric-random-epis_400_COMBINED.pkl
#done

# results_svg_3 [note the act-cond-2 at the end of the file name]

for MODEL in model_0050.pth model_0100.pth model_0200.pth model_0400.pth ; do
    python test_load_ssim.py \
        --model_dir=${HEAD}/fabric-random/model=dcgan56x56-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=3-n_future=7-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000-act-cond-2/${MODEL}  \
        --data_path=/home/seita/cloth-visual-mpc/logs/demos-fabric-random-epis_400_COMBINED.pkl
done


# ---------------------------------------- #
# FabricV2, 80-20 train-valid split
# ---------------------------------------- #

# results_svg_1

#for MODEL in model_0050.pth model_0100.pth model_0200.pth model_0400.pth model_0600.pth model_0800.pth model_1000.pth model_1200.pth model_1499.pth ; do
#    python predict_svg_lp.py \
#        --model_path=${HEAD}/fabric-01_2021/model=dcgan56x56-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=2-n_future=5-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000-act-cond-1/${MODEL}  \
#        --data_path=/home/seita/cloth-visual-mpc/logs/demos-fabric-01-2021-epis_400_COMBINED.pkl
#done

# results_svg_2

#for MODEL in model_0050.pth model_0100.pth model_0200.pth model_0400.pth model_0600.pth model_0800.pth model_1000.pth model_1200.pth model_1499.pth ; do
#    python test_load_ssim.py \
#        --model_dir=${HEAD}/fabric-01_2021/model=dcgan56x56-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=2-n_future=5-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000-act-cond-1/${MODEL}  \
#        --data_path=/home/seita/cloth-visual-mpc/logs/demos-fabric-01-2021-epis_400_COMBINED.pkl
#done

# results_svg_3 [note the act-cond-2 at the end of the file name]

for MODEL in model_0050.pth model_0100.pth model_0200.pth model_0400.pth model_0600.pth ; do
    python test_load_ssim.py \
        --model_dir=${HEAD}/fabric-01_2021/model=dcgan56x56-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=2-n_future=5-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000-act-cond-2/${MODEL}  \
        --data_path=/home/seita/cloth-visual-mpc/logs/demos-fabric-01-2021-epis_400_COMBINED.pkl
done
