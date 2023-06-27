import os
import glob
import h5py
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import argparse

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataDir',
        default=f"{os.environ['HOME']}/workspace/prj/data/SynthText",
        type=str,
        help="""
        This is the location of the background data.
        The data can the background data from here:
            https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c
        """
    )
    parser.add_argument(
        '--out',
        default='data/out-sample.h5',
        type=str,
        help='This is where you would like to save the sample file.'
    )
    return parser.parse_args()


# # This is the location of the background data.
# # The data can the background data from here:
# # https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c
# dataDir = f"{os.environ['HOME']}/workspace/prj/data/SynthText"

def get_valid_imnames(namesPath: str) -> list:
    assert os.path.isfile(namesPath)
    with open(namesPath, 'r') as f:
        lines = f.readlines()
    filenames = []
    for line in lines:
        line = line.replace('\n', '')
        basename, ext = os.path.splitext(line)
        if ext in ['', '.']:
            continue
        if ext not in ['.png', '.jpg', '.jpeg']:
            raise ValueError(f"Unexpected extension: {ext}. Line: {line}")
        if line.startswith('aV'):
            line = line[2:]
        elif line.startswith('V'):
            line = line[1:]
        else:
            raise ValueError(line)
        filenames.append(line)
    return filenames

def main(args: argparse.Namespace):
    if False:
        # Check what format we need to create.
        renderedDataDir = f"{dataDir}/renderer_data"
        sampleDsPath = f"{renderedDataDir}/sample.h5"
        assert os.path.isfile(sampleDsPath), f"File not found: {sampleDsPath}"
        sample = h5py.File(sampleDsPath, 'r')
        print(f"sample.keys(): {sample.keys()}")
        depth: h5py.Group = sample['depth']
        image: h5py.Group = sample['image']
        seg: h5py.Group = sample['seg']
        
        # Image data is organized like this.
        for key in image.keys():
            _img: h5py.Dataset = image[key]
            _depth: h5py.Dataset = depth[key]
            _seg: h5py.Dataset = seg[key]
            print(f"{key} _seg.attrs.keys(): {_seg.attrs.keys()}")
            _area: np.ndarray = _seg.attrs['area']
            _label: np.ndarray = _seg.attrs['label']
            print(f"{key} _img.shape: {_img.shape}")
            print(f"{key} _depth.shape: {_depth.shape}")
            print(f"{key} _seg.shape: {_seg.shape}")
            print(f"{key} _area.shape: {_area.shape}")
            print(f"{key} _label.shape: {_label.shape}")
    else:
        if os.path.isfile(args.out):
            os.remove(args.out)
        out_sample = h5py.File(args.out, 'w')

        assert os.path.isdir(args.dataDir), f"Failed to find directory: {args.dataDir}"
        bgDataDir = f"{args.dataDir}/bg_data"
        assert os.path.isdir(bgDataDir), f"Failed to find directory: {bgDataDir}"
        depthPath = f"{bgDataDir}/depth.h5"
        assert os.path.isfile(depthPath), f"Failed to find file: {depthPath}"
        segPath = f"{bgDataDir}/seg.h5"
        assert os.path.isfile(segPath), f"Failed to find file: {segPath}"
        bgImgDir = f"{bgDataDir}/bg_img"
        assert os.path.isdir(bgImgDir), f"Failed to find directory: {bgImgDir}"
        namesPath = f"{bgDataDir}/imnames.cp" # names of images which don't contain background text
        assert os.path.isfile(namesPath), f"Failed to find file: {namesPath}"

        validFilenames = get_valid_imnames(namesPath)
        for validFilename in validFilenames:
            validPath = f"{bgImgDir}/{validFilename}"
            assert os.path.isfile(validPath), f"File not found: {validPath}"

        depth = h5py.File(depthPath, 'r')
        seg = h5py.File(segPath, 'r')
        mask = seg['mask']

        imgPaths = []
        for extension in ['jpg', '.jpeg', '.png']:
            imgPaths += glob.glob(f"{bgImgDir}/*{extension}")
        imgPaths.sort()
        # print(len(imgPaths))

        out_sample.create_group('/image')
        out_sample.create_group('/depth')
        out_sample.create_group('/seg')

        for imgPath in tqdm(imgPaths):
            filename = os.path.basename(imgPath)
            if (
                not os.path.isfile(imgPath)
                or filename not in depth.keys()
                or filename not in mask.keys()
                or filename not in validFilenames
            ):
                continue
            try:
                img = Image.open(imgPath).convert('RGB')
            except Image.DecompressionBombError as e:
                continue
            img = np.array(img, dtype=np.uint8)
            _seg = mask[filename]
            _area: np.ndarray = _seg.attrs['area']
            _label: np.ndarray = _seg.attrs['label']
            out_sample['image'].create_dataset(filename, data=img)
            out_sample['depth'].create_dataset(filename, data=depth[filename])
            out_sample['seg'].create_dataset(filename, data=_seg)
            out_sample['seg'][filename].attrs['area'] = _area
            out_sample['seg'][filename].attrs['label'] = _label

            # _img: h5py.Dataset = out_sample['image'][filename]
            # _depth: h5py.Dataset = out_sample['depth'][filename]
            # _seg: h5py.Dataset = out_sample['seg'][filename]
            # print(f"{filename} _img.shape: {_img.shape}")
            # print(f"{filename} _depth.shape: {_depth.shape}")
            # print(f"{filename} _seg.shape: {_seg.shape}")
            # print(f"{filename} _area.shape: {_area.shape}")
            # print(f"{filename} _label.shape: {_label.shape}")

        out_sample.close()
        depth.close()
        seg.close()

if __name__ == '__main__':
    args = get_args()
    main(args)
