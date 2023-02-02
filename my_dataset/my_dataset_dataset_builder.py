"""my_dataset dataset."""

import io
import os
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
import numpy as np

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for geometrobj dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Register into https://example.org/login to get the data. Place the `data.zip`
  file in the `manual_dir/`.
  """


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(64, 64, 3)),
            'label': tfds.features.ClassLabel(names=['circle', 'rectangle', 'triangle']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://github.com/vnepogodin/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://matrix.cachyos.org')

    # data path
    archive_path = dl_manager.manual_dir / 'geometric_shapes_images.tar.gz'

    # Extract archive
    extracted_path = dl_manager.extract(archive_path)

    # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(extracted_path / 'geometric_shapes_images'),
    }

  def _generate_examples(self, images_dir_path):
    """Generate shapes images and labels given the image directory path.
    Args:
      images_dir_path: path to the directory where the images are stored.
    Yields:
      The image path and its corresponding label.
    """
    for fname in tf.io.gfile.glob(os.path.join(images_dir_path, '*', '*.jpg')):
     if fname.endswith(".jpg"):
       image_dir, image_file = os.path.split(fname)
       d = os.path.basename(image_dir)
       record = {"image": fname, "label": d.lower()}
       yield "%s/%s" % (d, image_file), record
