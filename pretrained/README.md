# Pretrained Checkpoints

This folder is for approved public ClassIC-SR checkpoints.

Default expected checkpoint path:

```text
pretrained/classic_sr_version_a.pth
```

For peer-review or public reproduction, place the approved ClassIC-SR checkpoint here. Without the checkpoint, users can run architecture profiling but cannot reproduce paper PSNR.

When the checkpoint is available through GitHub Releases or an approved supplementary review link, download it and place it at the default path:

```bash
mkdir -p pretrained
wget -O pretrained/classic_sr_version_a.pth https://github.com/YunlongLiu-code/ClassIC-SR/releases/download/v0.1.0/classic_sr_version_a.pth
```

If `wget` fails in a restricted network environment, use GitHub CLI as a fallback:

```bash
mkdir -p pretrained
gh release download v0.1.0 \
  --repo YunlongLiu-code/ClassIC-SR \
  --pattern classic_sr_version_a.pth \
  --dir pretrained
```

Do not commit private or unapproved checkpoints to this repository.
