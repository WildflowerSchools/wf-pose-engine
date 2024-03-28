import os
import sys
from urllib.parse import urlparse

import torch
from torch.hub import HASH_REGEX, _get_torch_home, download_url_to_file

from mmengine.utils.path import mkdir_or_exist


def download_url(
    url, checkpoint_dir=None, progress=True, check_hash=False, file_name=None
):
    r"""Loads the Torch serialized object at the given URL.
    If downloaded file is a zip file, it will be automatically decompressed
    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.
    Args:
        url (str): URL of the object to download
        checkpoint_dir (str, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar
            to stderr. Defaults to True
        check_hash(bool, optional): If True, the filename part of the URL
            should follow the naming convention ``filename-<sha256>.ext``
            where ``<sha256>`` is the first eight or more digits of the
            SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Defaults to False
        file_name (str, optional): name for the downloaded file. Filename
            from ``url`` will be used if not set. Defaults to None.
    Example:
        >>> url = ('https://s3.amazonaws.com/pytorch/models/resnet18-5c106'
        ...        'cde.pth')
        >>> state_dict = torch.hub.load_state_dict_from_url(url)
    """
    if checkpoint_dir is None:
        torch_home = _get_torch_home()
        checkpoint_dir = os.path.join(torch_home, "checkpoints")

    mkdir_or_exist(checkpoint_dir)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file
