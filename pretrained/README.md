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
# Replace the URL with the approved checkpoint asset URL.
wget -O pretrained/classic_sr_version_a.pth <APPROVED_CHECKPOINT_URL>
```

Do not commit private or unapproved checkpoints to this repository.
