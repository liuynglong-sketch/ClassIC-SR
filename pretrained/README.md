# Pretrained Checkpoints

This folder is for approved public ClassIC-SR checkpoints.

Default expected checkpoint path:

```text
pretrained/classic_sr_version_a.pth
```

The pretrained checkpoint will be released after paper acceptance. Without it, users can run architecture profiling but cannot reproduce paper PSNR.

When the checkpoint becomes available through GitHub Releases, download it and place it at the default path:

```bash
mkdir -p pretrained
# Replace the URL with the release asset URL once available.
wget -O pretrained/classic_sr_version_a.pth <RELEASE_ASSET_URL>
```

Do not commit private or unapproved checkpoints to this repository.
