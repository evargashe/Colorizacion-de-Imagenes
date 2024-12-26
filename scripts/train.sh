DATASET_DIR=train_data/train10000

# Stage 1: Training Full Image Colorization
python train.py --stage full --name coco_full --sample_p 1.0 --niter 15 --niter_decay 5 --load_model --lr 0.0005 --model train --fineSize 256 --batch_size 2 --display_ncols 3 --display_freq 1600 --print_freq 1600 --train_img_dir train_data/train10000

# Stage 2: Training Instance Image Colorization
python train.py --stage instance --name coco_instance --sample_p 1.0 --niter 15 --niter_decay 5 --load_model --lr 0.0005 --model train --fineSize 256 --batch_size 2 --display_ncols 3 --display_freq 1600 --print_freq 1600 --train_img_dir train_data/train10000

# Stage 3: Training Fusion Module
python train.py --stage fusion --name coco_mask --sample_p 1.0 --niter 10 --niter_decay 5 --lr 0.00005 --model train --load_model --display_ncols 4 --fineSize 128 --batch_size 1 --display_freq 500 --print_freq 500 --train_img_dir train_data/train10000
