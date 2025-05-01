# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import argparse
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lo.sdk.api.acquisition.io.open import _guess_filetype
from lo.sdk.api.acquisition.io.open import open as sdk_open
from lo.sdk.api.acquisition.io.wrappers.lofmt import LOFmtIO


def reflectance_to_radiance(
    reflectance: np.ndarray,
    black_spectrum=None,
    white_spectrum=None,
):
    """
    Converts reflectance to radiance using provided white/black spectrum or auto-estimation.

    Args:
        radiance (np.ndarray): spectral list (N x C)
        black_spectrum (np.ndarray, optional): black reference spectrum
        white_spectrum (np.ndarray, optional): white reference spectrum

    Returns:
        radiance (np.ndarray): spectral list (N x C)
    """
    if white_spectrum is None or black_spectrum is None:
        raise ValueError("Both white_spectrum and black_spectrum must be provided.")

    radiance = reflectance * (white_spectrum - black_spectrum) + black_spectrum

    return radiance


def radiance_to_reflectance(
    radiance: np.ndarray,
    black_spectrum: np.ndarray = None,
    white_spectrum: np.ndarray = None,
    q: float = 95.0,
):
    """
    Converts radiance to reflectance using provided white/black spectrum or auto-estimation.

    Args:
        radiance_spectral_list (np.ndarray): spectral list (N x C)
        black_spectrum (np.ndarray, optional): black reference spectrum
        white_spectrum (np.ndarray, optional): white reference spectrum
        q (float, optional): Percentile for auto white estimation if not provided.

    Returns:
        reflectance (np.ndarray): spectral list (N x C)
    """
    if white_spectrum is None:
        white_spectrum = get_illuminant(radiance, q)
    if black_spectrum is None:
        black_spectrum = np.zeros_like(white_spectrum)

    reflectance = (radiance - black_spectrum) / (white_spectrum - black_spectrum)

    return reflectance


def get_illuminant(
    radiance: np.ndarray,
    q: float = 95.0,
):
    """
    Does automatic reflectance conversion on a list of spectra by using the specified
    method to select the white reference spectra.

    Args:
        radiance_spectral_list (xp.ndarray): spectral list (N x C)
        q (int, optional): Percentile of brightest pixels to average over as the
            white reference spectrum for the percentile method. Defaults to 95.

    Returns:
        white_spectrum (xp.ndarray): the estimated illuminant
    """

    sumi = np.sum(radiance, axis=1)
    percentile = np.percentile(sumi, q=q)
    mask = sumi > percentile
    white_spectrum = np.mean(radiance[mask], axis=0)

    return white_spectrum


def rle_to_mask(mask_rle: str, shape: Tuple[int, int] = (2048, 2432), label: int = 1):
    """
    Convert an RLE string to a mask array.
    Args:
        mask_rle (str): run-length as string formatted (start length)
        shape (Tuple[int, int]):
        label (int): class label for foreground pixels of mask (Default = 1)

    Returns:
        mask (np.ndarray): decoded RLE string as a mask. {label} as foreground, 0 as background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    if len(starts) == len(lengths) + 1:
        total = shape[0] * shape[1]
        missing_length = total - starts[-1]
        lengths = np.append(lengths, missing_length)
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = label
    return np.reshape(mask, shape).astype(np.uint8)  # Needed to align to RLE direction


def spectral_coordinate_indices_in_mask(mask: np.ndarray, sampling_coordinates: np.ndarray) -> np.ndarray:
    """
    Returns an array of indices of sampling_coordinates that fall inside the mask.

    Parameters:
    mask (np.ndarray): A boolean 2D array where True values indicate valid regions.
    sampling_coordinates (np.ndarray): An (N, 2) array of (row, col) coordinates.

    Returns:
    np.ndarray: Indices of sampling_coordinates that fall inside the mask.
    """
    sampling_coordinates = np.int32(sampling_coordinates)
    in_mask = mask[sampling_coordinates[:, 0], sampling_coordinates[:, 1]]
    return np.where(in_mask)[0]


class DatasetProcessor:
    def __init__(
        self,
        dataset_path: str,
        display_fig: bool = False,
        save_fig: bool = True,
        fig_path: str = "visualisations",
        unit_conversion: bool = False,
    ):
        """
        Initialize the dataset processor.

        Args:
            dataset_path: Path to the exported dataset folder
            display_fig: Whether to display figures interactively
            save_fig: Whether to save figures to disk
            fig_path: Path to save figures
        """
        self.dataset_path = dataset_path
        self.display_fig = display_fig
        self.save_fig = save_fig
        self.fig_path = fig_path
        self.unit_conversion = unit_conversion

        self.dataset = DatasetReader(
            dataset_path=self.dataset_path,
            display_fig=self.display_fig,
            save_fig=self.save_fig,
            fig_path=self.fig_path,
            unit_conversion=self.unit_conversion,
        )
        self.library_spectra = self.dataset.library_spectra

    def print_metadata(self, images_extern: Dict):
        """Print images_extern information"""
        print(f"Frame number: {images_extern['frame_number']}")
        print(f"Image shape: ({images_extern['width']}, {images_extern['height']})")
        print(f"License url: {images_extern['acquisition']['license_url']}")
        print(f"White spectrum id: {images_extern['white_spectrum_id']}")
        print(f"Black spectrum id: {images_extern['black_spectrum_id']}")
        print(f"Total number of frames: {images_extern['acquisition']['total_frames']}")

    def process_annotation(self, scene: np.ndarray, annotation: Dict, images_extern: Dict, ann_idx: int, idx: int):
        """Process a single annotation"""
        if self.save_fig or self.display_fig:
            self.dataset.save_annotation_visualisation(scene, annotation, images_extern, ann_idx)

        self._print_annotation_details(annotation)

    def summarize_spectra(self):
        """Summarize all spectra in the dataset"""
        df = self.dataset.list_all_spectra(as_dataframe=True)
        print("\nComplete spectra library summary:")
        print(df.to_string(index=False))

        for spectrum_name in self.dataset.list_all_spectra():
            spectrum_data = self.dataset.get_spectrum(spectrum_name)
            if spectrum_data and self.save_fig:
                self._plot_target_spectrum(spectrum_data)

    def process_dataset(self):
        """Main method to process the entire dataset"""
        os.environ["QT_QPA_PLATFORM"] = "xcb"

        for idx, (
            (info, scene, spectra, images_extern),
            converted_spectra,
            annotations,
            library_spectra,
            labels,
        ) in enumerate(self.dataset):
            self._process_frame(scene, images_extern, annotations, labels, idx)

        self.summarize_spectra()

    def _print_annotation_details(self, annotation: Dict):
        """Print details of an annotation"""
        print("\n--- Annotation details ---")
        print(f"Bounding box: {annotation['bbox']}")
        print(f"Category: {annotation['category_name']}")
        print(
            f"Date created: {datetime.fromisoformat(annotation['extern']['date_created']).replace(microsecond=0).isoformat()}"
        )
        print(f"Labelling method: {annotation['extern']['labelling_method']}")
        print(f"Area: {annotation['extern']['area']}")
        for i in range(len(annotation["metadata"])):
            print(f"User data({annotation['metadata'][i]['field']}): {annotation['metadata'][i]['value']}")

    def _plot_target_spectrum(self, spectrum_data: Tuple[str, str, str, np.ndarray, str, str]):
        """Plot a target spectrum"""
        name, target, unit, data, type, id = spectrum_data

        fig = plt.figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        unit_line = f"Unit: {unit}"
        type_line = f"Type: {type}"

        plt.plot(data[:, 0], data[:, 1])
        plt.title(f"Target Spectrum: {name}")
        ax.annotate(
            unit_line,
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=10,
            ha="left",
            va="top",
            color="white",
            bbox=dict(facecolor="blue", alpha=0.7, boxstyle="round,pad=0.3"),
        )

        ax.annotate(
            type_line,
            xy=(0.02, 0.93),
            xycoords="axes fraction",
            fontsize=10,
            ha="left",
            va="top",
            color="white",
            bbox=dict(facecolor="blue", alpha=0.7, boxstyle="round,pad=0.3"),
        )

        plt.xlabel("Wavelength")
        plt.ylabel(unit)
        plt.tight_layout()

        if self.save_fig:
            save_path = os.path.join(self.fig_path, "target_spectrum")
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"target_spectrum_{name}.png"), pad_inches=0)
        if self.display_fig:
            plt.show()
        else:
            plt.close()

    def _process_frame(
        self, scene: np.ndarray, images_extern: Dict, annotations: List[Dict], labels: List[str], idx: int
    ):
        """Process a single frame"""
        file_name = images_extern.get("acquisition", {}).get("raw_file_name", "").split(".")[0]
        print(f"\n=== Processing file: {file_name} frame {idx} with {len(annotations)} annotations ===")

        if not annotations:
            print("No annotated item found in frame")
            return

        self.print_metadata(images_extern)

        for ann_idx, annotation in enumerate(annotations):
            annotation["labels"] = labels
            self.process_annotation(scene, annotation, images_extern, ann_idx, idx)


class DatasetReader:
    def __init__(
        self,
        dataset_path: str,
        display_fig: bool = False,
        save_fig: bool = True,
        fig_path: str = "visualisations",
        unit_conversion: bool = False,
    ):
        """
        Initialize the dataset reader.

        Args:
            dataset_path: Path to the exported dataset folder
            units: Desired output units for spectra
            display_fig: Whether to display figures
            save_fig: Whether to save figures
            fig_path: Path to save figures
        """
        self.base_path = Path(dataset_path)
        self.zip_file = None
        if dataset_path.endswith(".zip"):
            self.zip_file = zipfile.ZipFile(dataset_path, "r")
            self.data_path = "data"
            self.json_data = self._load_metadata_zip()
        else:
            self.base_path = Path(dataset_path)
            self.data_path = self.base_path / "data"
            self.json_data = self._load_metadata_file()
        self.display_fig = display_fig
        self.save_fig = save_fig
        self.fig_path = fig_path
        self.unit_conversion = unit_conversion

        if self.save_fig:
            os.makedirs(self.fig_path, exist_ok=True)

        # Load metadata and build indices
        self.library_spectra = self._build_library_spectra()
        self.category_map = self._build_category_map()
        self.frames = self._index_frames()

    def _load_metadata_file(self) -> Dict:
        metadata_path = self.base_path / "extended_coco.json"
        with open(metadata_path, "r") as f:
            return json.load(f)

    def _load_metadata_zip(self) -> Dict:
        json_files = [f for f in self.zip_file.namelist() if f.endswith("extended_coco.json")]
        if not json_files:
            raise FileNotFoundError("extended_coco.json not found in zip archive.")
        with self.zip_file.open(json_files[0]) as f:
            return json.load(f)

    def __iter__(self) -> Iterator[Tuple]:
        """Iterate over the dataset yielding (scene, spectra, metadata), annotations, labels"""
        for frame in self.frames:
            file_path = os.path.join(self.data_path, frame["file_name"])
            b_v = None  # black_spectrum_value
            w_v = None  # white_spectrum_value
            converted_spectra = None  # if unit_conversion

            try:
                info, scene, spectra = self._read_lo_frame(str(file_path))

                if self.unit_conversion:
                    for k, v in frame["extern"].items():
                        unit = v["units"]
                        if v["id"] == frame["image_extern"]["black_spectrum_id"]:
                            b_v = v["values"]
                        elif v["id"] == frame["image_extern"]["white_spectrum_id"]:
                            w_v = v["values"]
                    if unit == "radaiance":
                        converted_spectra = radiance_to_reflectance(
                            radiance=spectra, black_spectrum=b_v, white_spectrum=w_v
                        )
                    if unit == "reflectance":
                        converted_spectra = reflectance_to_radiance(
                            reflectance=spectra, black_spectrum=b_v, white_spectrum=w_v
                        )

                yield (
                    (info, scene, spectra, frame.get("metadata", frame.get("image_extern", {}))),
                    converted_spectra,
                    frame["annotations"],
                    frame["extern"],
                    [a.get("category", a.get("category_name", "unknown")) for a in frame["annotations"]],
                )
            except Exception as e:
                print(f"Warning: Could not read frame {file_path}: {str(e)}")
                continue

    def get_spectrum(self, name: str) -> Optional[Tuple[str, str, np.ndarray]]:
        """
        Get spectrum by name in standardized format.

        Returns:
            Tuple of (name, unit, Nx2 array) or None if not found
        """
        spec = self.library_spectra.get(name)
        if not spec:
            return None

        return (
            name,
            spec["target"],
            spec["units"],
            np.column_stack((spec["wavelengths"], spec["values"])),
            spec["spectrum_type"],
            (spec["white_spectrum_id"], spec["black_spectrum_id"]),
        )

    def list_all_spectra(self, as_dataframe: bool = False) -> Union[List[str], pd.DataFrame]:
        """
        List available spectra.

        Args:
            as_dataframe: If True, returns DataFrame with full summary
        """
        if not as_dataframe:
            return list(self.library_spectra.keys())

        spectra_data = []
        for name, spec in self.library_spectra.items():
            spectra_data.append(
                {
                    "name": name,
                    "spectrum_type": spec["spectrum_type"],
                    "units": spec["units"],
                    "source": spec["source_format"],
                    "n_points": len(spec["values"]),
                    "wavelength_range": f"{np.min(spec['wavelengths']):.1f}-{np.max(spec['wavelengths']):.1f} nm",
                    "value_range": f"{np.min(spec['values']):.4f}-{np.max(spec['values']):.4f}",
                    "white_spectrum_id": spec["white_spectrum_id"],
                    "black_spectrum_id": spec["black_spectrum_id"],
                }
            )
        return pd.DataFrame(spectra_data)

    def save_annotation_visualisation(
        self, scene: np.ndarray, annotation: Dict, images: Dict, ann_idx: int
    ) -> np.ndarray:
        """
        Save visualisation of an annotation.

        Returns:
            Generated mask (if segmentation exists)
        """
        file_name = images.get("acquisition", {}).get("raw_file_name", "").split(".")[0]
        fig = plt.figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        ax.imshow(scene, cmap="gray")

        # Draw bounding box
        bbox = annotation["bbox"]
        x, y, w, h = bbox
        rect = plt.Rectangle((y, x), h, w, linewidth=1, edgecolor="b", facecolor="none")
        ax.add_patch(rect)

        # Draw segmentation if available
        mask = None
        if annotation["segmentation"]:
            mask = rle_to_mask(annotation["segmentation"], scene.shape)
            ax.imshow(mask, alpha=0.3, cmap="jet")

        ax.set_title(f"Annotation: {annotation['category_name']}")

        if annotation["metadata"]:
            for idx, metadata in enumerate(annotation["metadata"]):
                metadata_line = f"{metadata['field']}: {metadata['value']}"

                y_position = 0.98 - (idx * 0.05)

                ax.annotate(
                    metadata_line,
                    xy=(0.98, y_position),
                    xycoords="axes fraction",
                    fontsize=10,
                    ha="right",
                    va="top",
                    color="white",
                    bbox=dict(facecolor="blue", alpha=0.7, boxstyle="round,pad=0.3"),
                )

        ax.axis("off")
        plt.tight_layout()

        if self.save_fig:
            save_path = os.path.join(self.fig_path, "annotations")
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                os.path.join(save_path, f"{file_name}_{annotation['extern']['category']}_{ann_idx}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
        if self.display_fig:
            plt.show()
        else:
            plt.close()

        return mask

    def _load_metadata(self) -> Dict:
        """Load metadata from extended_coco.json"""
        json_path = self.base_path / "extended_coco.json"
        with open(json_path, "r") as f:
            return json.load(f)

    def _build_library_spectra(self) -> Dict[str, Dict]:
        """Build a lookup of all spectra in the dataset"""
        library = {}
        if self.json_data.get("extern", {}).get("spectra", {}):
            for spectrum in self.json_data["extern"]["spectra"]:
                library[spectrum["field"]] = {
                    "id": spectrum["id"],
                    "source_format": spectrum["source_format"],
                    "target": spectrum["field"],
                    "units": spectrum["units"],
                    "wavelengths": np.array(spectrum["wavelengths"]),
                    "values": np.array(spectrum["values"]),
                    "spectrum_type": spectrum["spectrum_type"],
                    "white_spectrum_id": spectrum["white_spectrum_id"],
                    "black_spectrum_id": spectrum["black_spectrum_id"],
                }
        return library

    def _build_category_map(self) -> Dict[int, str]:
        """Build mapping from category IDs to names"""
        return {c["id"]: c["name"] for c in self.json_data.get("categories", [])}

    def _index_frames(self) -> List[Dict]:
        """Index all image frames with full metadata and annotation details"""
        indexed_frames = []

        for image in self.json_data.get("images", []):
            frame_entry = {
                "id": image["id"],
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "frame_number": image.get("frame_number"),
                "license": image.get("license"),
                "date_captured": image.get("date_captured"),
                "image_extern": image.get("extern", {}),
                "acquisition": image.get("extern", {}).get("acquisition", {}),
                "annotations": [],
                "extern": self.library_spectra,
            }

            for annotation in self.json_data.get("annotations", []):
                if annotation["image_id"] == image["id"] and annotation["image_id"] == annotation["extern"]["frame_id"]:
                    annotation_entry = {
                        "id": annotation["id"],
                        "category_id": annotation["category_id"],
                        "bbox": annotation["bbox"],
                        "area": annotation.get("area"),
                        "segmentation": annotation.get("segmentation"),
                        "iscrowd": annotation.get("iscrowd"),
                        "extern": annotation.get("extern", []),
                        "metadata": annotation.get("metadata", []),
                        "category_name": self.category_map.get(annotation["category_id"], "unknown"),
                    }

                    frame_entry["annotations"].append(annotation_entry)

            indexed_frames.append(frame_entry)

        return indexed_frames

    def _get_target_spectra(self, annotation: Dict) -> Optional[Dict]:
        """Get target spectra from annotation if exists"""
        if annotation["metadata"]["field"] == "position":
            target_name = annotation["metadata"]["value"]
            if target_name in self.library_spectra:
                return {"name": target_name, **self.library_spectra[target_name]}
        return None

    def _read_lo_frame(self, file_name: str) -> Tuple[Dict, np.ndarray, np.ndarray]:
        dir_name, zip_format = os.path.splitext(os.path.basename(self.base_path))
        if self.zip_file:
            file_path = os.path.join(dir_name, file_name)
            with self.zip_file.open(file_path) as zipped_file:
                raw_bytes = zipped_file.read()
                with tempfile.NamedTemporaryFile(delete=True, suffix=".lo") as temp_file:
                    temp_file.write(raw_bytes)
                    temp_file.flush()  # Write to disk
                    with sdk_open(temp_file.name) as lo_file:
                        info, scene, spectra = lo_file.read()
                        return info, scene.squeeze(), spectra
        else:
            """Read a single frame from LO file"""
            filetype = _guess_filetype(file_name)
            if filetype != LOFmtIO:
                raise ValueError(f"Unsupported file type: {filetype}")

            with sdk_open(file_name) as lo_file:
                info, scene, spectra = lo_file.read()
                return info, scene.squeeze(), spectra


def main():
    default_figures_path = "visualisations"

    parser = argparse.ArgumentParser(
        description="Process Living Optics dataset and generate visualisations.",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the exported dataset folder",
    )
    parser.add_argument(
        "--display-figures",
        action="store_true",
        help="Display figures interactively",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        default=True,
        help="Save figures to disk",
    )
    parser.add_argument(
        "--figures-path",
        type=str,
        default=default_figures_path,
        help="Path to save figures",
    )
    parser.add_argument("--unit-conversion", action="store_true", default=False, help="Perform unit conversion")
    args = parser.parse_args()

    if args.figures_path == default_figures_path:
        base = os.path.splitext(args.path)[0]
        parent = os.path.dirname(base)
        name = os.path.basename(base)
        args.figures_path = os.path.join(parent, f"{name}_visualisations")

    processor = DatasetProcessor(
        dataset_path=args.path,
        fig_path=args.figures_path,
        display_fig=args.display_figures,
        save_fig=args.save_figures,
        unit_conversion=args.unit_conversion,
    )
    processor.process_dataset()


if __name__ == "__main__":
    main()
