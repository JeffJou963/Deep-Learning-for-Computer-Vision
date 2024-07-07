python -c "import clip; clip.load('ViT-B/32')"

python -c "import timm; timm.create_model('vit_gigantic_patch14_clip_224', pretrained=True)"

wget -O CIDER0.8410.ckpt https://www.dropbox.com/scl/fi/tbbe4b09relw6bjzgii15/CIDER0.8410.ckpt?rlkey=lyq7j6xpowuabb4xho2h4swmz&dl=1
