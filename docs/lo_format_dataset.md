# COCO Extended Format Documentation (Based on `extended_coco.json`)

## Dependencies & Tools for COCO Extended Dataset

This document outlines the essential dependencies and tools required to load
and visualise datasets exported in the Living Optics COCO Extended format.

---

### Essential Dependencies

| Library        | Description                                         | Installation Command             |
|----------------|-----------------------------------------------------|----------------------------------|
| **lo_sdk**     | Official SDK for Living Optics. Used for opening `.lo` files and parsing spectra, frames, etc. | `pip install ./lo_sdk-{version}-py3-none-any.whl`  # get latest SDK wheel           |
| **pandas**     | Used for summarising spectrum tables in DataFrame format. | `pip install pandas`             |
| **matplotlib** | Basic plotting library for visualisations.          | `pip install matplotlib`         |

> `lo_sdk` is used through `lo.sdk.api.acquisition.io.open` and
`.wrappers.lofmt` to handle `.lo` format files.

---

### Utility Script: `lo_dataset_reader.py`

`lo_dataset_reader.py` is the core tool used to load and analyse Living
Optics datasets exported in the COCO Extended format.

#### Key Features

- Automatic loading of `extended_coco.json`
- Frame-by-frame reading of `.lo` files
- Output of metadata including `white_spectrum_id` and `black_spectrum_id`
- Visualisation of bounding boxes and segmentations
- Extraction and plotting of target spectra
- Tabular summary of library spectra


#### Example Usage

```bash
python lo_dataset_reader.py --path /path/to/dataset --save-figures
```

### white_spectrum_id, black_spectrum_id Fields

- These fields are used for normalising each frame or spectrum.
- They are included in the extern field of images[] or spectra[] in the extended_coco.json.

### Example Visualisation Outputs

- `exported_group_visualisations/annotations/grape_frame_0.png` : Annotation bounding box + segmentation
- `exported_group_visualisations/target_spectrum/target_spectrum_tray-tray2.png` : Spectrum plot for the white/black spectrum

## Folder Structure

```plaintext
Exported Group/
├── data/         # Hyperspectral imaging data (.lo files)
├── thumbnails/         # Visual representations of .lo files (.png)
└── extended_coco.json   # COCO-based annotation file (with extensions)
```

---

## JSON Structure Overview

This dataset extends the standard COCO format to support `.lo` files. It
retains COCO’s core components (info, images, annotations, categories) and
introduces an `extern` structure to provide additional metadata tailored for
spectral imaging.

### info

| Field           | Type    | Description                            |
|-----------------|---------|----------------------------------------|
| description     | string  | Description of the exported group      |
| version         | string  | Version information                    |
| year            | int     | Year of creation                       |
| contributor     | string  | Contributor name                       |
| date_created    | string  | Date of creation (ISO 8601 format)     |

---

### licenses

| Field  | Type   | Description                          |
|--------|--------|--------------------------------------|
| id     | int    | Licence ID                           |
| name   | string | Name of the licence                  |
| url    | string | URL to the licence                   |

---

### images

This follows the standard COCO structure but includes additional metadata for
`.lo` files via the `extern` field. The image section links to the original
data (`.lo` files), image dimensions, frame information, and acquisition
context.

- `image["id"]` : A unique identifier for each image object in the exported
group. This field is used internally to match images to their corresponding
metadata and annotations.

| Field          | Type      | Description                                  |
|----------------|-----------|----------------------------------------------|
| id             | string    | UUID of the image                            |
| width          | int       | Image width                                  |
| height         | int       | Image height                                 |
| file_name      | string    | Name of the `.lo` file                       |
| frame_number   | int       | Frame number                                 |
| license        | int       | Licence ID                                   |
| date_captured  | string    | Capture date (UNIX timestamp style)          |
| extern         | object    | Extended metadata (see example below)      |

#### Example structure for `extern` field (image-level metadata)

- `white_spectrum_id` *(string, optional)*
  - ID of the white reference spectrum used during normalisation. Typically
  derived from a high-reflectance target like Tyvek or Spectralon.
  - Represents the *maximum* reflectance across wavelengths.

- `black_spectrum_id` *(string, optional)*
  - ID of the black (dark current) reference spectrum used during normalisation.
  - Represents the *minimum* signal, ideally from a light-blocked or shaded image.

- The `white_spectrum_id` and `black_spectrum_id` fields refer to entries in
the library spectra list. This list stores various reference spectra—white,
black, and target—that are linked by their unique id.

The standard formula for reflectance correction:

```python
corrected = (raw - black) / (white - black)
```

```json
{
  "date_updated": str,
  "id": str,
  "frame_number": int,
  "height": int,
  "calibration_frame_id": str,
  "black_spectrum_id": str,
  "date_created": str,
  "acquisition_id": str,
  "width": int,
  "calibration_id": str,
  "white_spectrum_id": str,
  "acquisition": {
    "id": str,
    "date_captured": str,
    "file_name": str,
    "total_frames": int,
    "camera_id": str,
    "license_url": str
    "raw_file_name": str,
    "date_created": str,
    "lo_url": str,
    "date_updated": str,
    "format": str,
  }
}
```

---

### annotations

This section includes standard COCO annotation fields and introduces
additional spectral metadata for each object instance.

- `annotation["id"]`: A unique identifier for each annotation object. Used as a
stable reference to identify a labelled object within an image.

| Field         | Type      | Description                                      |
|---------------|-----------|--------------------------------------------------|
| id            | string    | Annotation UUID                                  |
| image_id      | string    | Referenced image UUID                            |
| category_id   | int       | Class ID (from categories)                       |
| bbox          | [float]   | Bounding box as [x, y, width, height]            |
| area          | float     | Object area (pixels^2)                           |
| segmentation  | string    | RLE-style encoded segmentation                   |
| iscrowd       | int       | Crowd flag (0 or 1)                              |
| extern        | object    | Extended annotation metadata                     |
| metadata      | [object]  | Additional labelled info (see below)             |

#### Example structure for `extern` field (annotation-level metadata)

```json
{
    "frame_id": str,
    "date_created": str,
    "supercategories": str,
    "area": float,
    "segmentation": RLE,
    "labelling_method": str,
    "class_name": str,
    "date_updated": str,
    "id": str,
    "category": str,
    "subcategories": [str],
    "bbox": [x,y,width,height],
    "spectral_segmentation_indexes": [int],
    "instance_id": int,
    "class_number": int
}
```

#### Example structure for `metadata` field

- `field` : A user-defined string that identifies the type or category of the metadata.
Users are free to define their own field names depending on the context and examples.  Accessed via annotation["metadata"]["field"].
value key in this metadata object corresponds to spectrum["field"].
- `value` : The content associated with the given `field`. The type of value
(string, number, etc.) depends on the definition and usage of the field.

```json
[
    {
        "parent_id": str,
        "date_created": str,
        "source_id": str,
        "field": str,
        "values": str | float,
        "targets": str,
        "target_mapping": str,
        "theoretical_min": float,
        "stats_method": str,
        "parent_type": str,
        "date_updated": str,
        "id": str,
        "source_type": str,
        "value": str,
        "target": str,
        "method_name": str,
        "theoretical_max": float,
        "source_names": str
    }
]
```

---

### categories

| Field            | Type   | Description               |
|------------------|--------|---------------------------|
| id               | int    | Category ID               |
| name             | string | Category name             |
| supercategory    | string | Higher-level grouping     |

---

### extern (Global Metadata)

This field captures global dataset-level information, such as spectra used for processing.

| Field             | Type      | Description                                       |
|-------------------|-----------|---------------------------------------------------|
| migration_version | string    | Version of the format or migration tag            |
| spectra           | [object]  | List of objects consists of additional metadata   |

#### Example structure for `spectra` field

- `units` : Describes the nature of the `values` array within a spectrum
object. Can be one of:
  - `reflectance`: Unitless ratio (0–1). Represents the proportion of incident
  light reflected by a surface. The data is normalised using white and black
  references and commonly used in comparative spectral analysis.
  - `radiance` : Physical intensity values (e.g., μW/cm²/nm/sr). Represents the
  total spectral energy captured by the sensor, possibly corrected for
  calibration, but not normalised to any reference.
  - `spectral_radiance`: Wavelength-resolved radiometric data. Similar to
  radiance, but explicitly indicates per-wavelength intensity values.
  Typically used as raw inputs for further calibration (e.g., white/black
  reference correction or target analysis).

- `spectrum_type`: Indicates the functional role of the spectrum.
Can be oneof:
  - `white` : Reference spectrum representing maximum reflectance
  (e.g., Tyvek).
  - `black` : Baseline or dark current reference
  (e.g., shadow or sensor dark signal).
  - `target` (or custom labels such as `erbium`, `sample`, etc.) : Spectrum
  from a subject of interest, typically measured for analysis or calibration.
  - `unknown` : Used when the origin or purpose of the spectrum is unclear.
  - (Optionally) `denoised` or `interpolated` can be used to indicate spectra
  processed through smoothing, noise reduction, or interpolation methods.

- `values` : Array of spectral intensity measurements. Each value corresponds
positionally to a wavelength. Interpretation depends on the `units`.

- `wavelengths` : Array of wavelengths (in nanometres) corresponding to each
value in values. Defines the spectral axis.

- `field` : Descriptive label for the type or context of the spectrum.
Examples:
  - `dark-reference-...` :  Dark or black reference
  - `white-reference-...` : White calibration (e.g., Tyvek)
  - `denoised-sample-...`, `erbium-reference-...` : Sample spectra or material
  references with or without processing

- `white_spectrum_id` : Optional ID linking to a white reference spectrum.
Useful for reviewing or reapplying normalisation.

- `black_spectrum_id` : Optional ID linking to a black (dark) reference. Used
for correcting sensor baseline or noise.

```json
[ls
    {
        "date_updated": str,
        "id": str,
        "parent_type": str,
        "field": str,
        "wavelengths": [float],
        "source_format": str,
        "white_spectrum_id": str,
        "date_created": str,
        "parent_id": str,
        "spectrum_type": str,
        "values": [float],
        "units": str,
        "source_file": str,
        "black_spectrum_id": str
    }
]
```

---

## Data Linking Overview

- Each annotation's metadata links to spectra based on specific fields and
values:

```plaintext
                ↓
      annotation["metadata"]["field"] == "target_spectra"
                ↓
        annotation["metadata"]["value"] == spectrum["field"]
                ↓
     spectrum["values"], spectrum["wavelengths"], spectrum["units"]
```

---

## Summary of Extensions

| Feature                         | Description                                                         |
|---------------------------------|---------------------------------------------------------------------|
| Support for `.lo` image format  | Enables the use of hyperspectral image files in .lo format, which contain spectral data.                        |
| `extern` metadata structure     | Provides a standardized metadata format present in both the image and annotation sections, offering additional context for each object and spectral data link.                       |
| Spectral segmentation indices   | Contains indices to spectral samples which lie within the annotated region.                       |
| Flexible metadata               | Allows the addition of custom metadata fields within annotations, enabling enhanced flexibility for storing diverse information.                      |
| Spectra storage                 | Stores spectral information, including wavelength, intensity, and unit data, ensuring that all relevant spectral data is linked correctly for analysis.   |

---

## References

- COCO Official Format: [https://cocodataset.org/#format-data](https://cocodataset.org/#format-data)
- `.lo` File Format: Spatial-spectral data format containing video recordings
from the Living Optics Camera
