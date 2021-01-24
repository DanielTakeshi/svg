# Perform model predictions.
HEAD=/data/svg/logs-Jan-24-SVG-1500epochs

# ---------------------------------------- #
# dold (old data), 80-20 train-valid split #
# ---------------------------------------- #

for MODEL in model_0050.pth model_0100.pth model_0200.pth model_0500.pth model_1000.pth model_1499.pth ; do
    python predict_svg_lp.py \
        --model_path=${HEAD}/fabric-random/model=dcgan56x56-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=3-n_future=7-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000-act-cond-1/${MODEL}  \
        --data_path=/home/seita/cloth-visual-mpc/logs/demos-fabric-random-epis_400_COMBINED.pkl
done

# ---------------------------------------- #
# dnew (new data), 80-20 train-valid split #
# ---------------------------------------- #

for MODEL in model_0050.pth model_0100.pth model_0200.pth model_0500.pth model_1000.pth model_1499.pth ; do
    python predict_svg_lp.py \
        --model_path=${HEAD}/fabric-01_2021/model=dcgan56x56-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=2-n_future=5-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000-act-cond-1/${MODEL}  \
        --data_path=/home/seita/cloth-visual-mpc/logs/demos-fabric-01-2021-epis_400_COMBINED.pkl
done
