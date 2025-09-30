import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import MONAI components with fallbacks
try:
    import monai
    from monai.transforms import Compose, LoadImage, ScaleIntensity, Resize, ToTensor
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("MONAI not available. Using basic image processing.")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using rule-based analysis.")

class BrainFindingsExtractor:
    def __init__(self):
        """Initialize the simplified brain findings extractor.
          find_dicom_series --> find all DICOM series in a folder.
          convert_series_to_nifti --> convert a specific DICOM series to NIfTI format.
          load_nifti_image --> load NIfTI image and return image data and header.
          basic_brain_segmentation --> basic brain segmentation using intensity thresholding.
          analyze_findings --> analyze brain volumes and derive clinical findings.
          get_severity --> determine severity based on thresholds.
          generate_clinical_summary --> generate a clinical summary based on findings.
          process_dicom_folder --> complete pipeline: DICOM → NIfTI → Analysis → Findings for all series.
          process_dicom_file --> process a single DICOM file.
          process_dicom_files --> process multiple DICOM files.
          save_results --> save the results to a JSON file.
          load_results --> load results from a JSON file.
          print_results --> print the results in a human-readable format
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        print(f"Using device: {self.device if self.device else 'CPU (no GPU/torch)'}")

    def find_dicom_series(self, dicom_folder):
        """Find all DICOM series in a folder."""
        try:
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(dicom_folder)

            if not series_ids:
                print(f"No DICOM series found in {dicom_folder}")
                return []

            series_info = []
            for series_id in series_ids:
                series_files = reader.GetGDCMSeriesFileNames(dicom_folder, series_id)
                if series_files:
                    # Get series description if available
                    try:
                        reader.SetFileNames(series_files)
                        reader.MetaDataDictionaryArrayUpdateOn()
                        reader.LoadPrivateTagsOn()

                        # Try to get series description
                        series_desc = "Unknown"
                        if reader.HasMetaDataKey(0, "0008|103e"):  # Series Description
                            series_desc = reader.GetMetaData(0, "0008|103e")

                        series_info.append({
                            'series_id': series_id,
                            'series_description': series_desc,
                            'num_files': len(series_files),
                            'files': series_files
                        })
                    except:
                        series_info.append({
                            'series_id': series_id,
                            'series_description': f"Series_{series_id}",
                            'num_files': len(series_files),
                            'files': series_files
                        })

            return series_info

        except Exception as e:
            print(f"Error finding DICOM series: {e}")
            return []

    def convert_series_to_nifti(self, series_files, output_path):
        """Convert a specific DICOM series to NIfTI format."""
        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(series_files)
            image = reader.Execute()

            # Get image information
            spacing = image.GetSpacing()
            size = image.GetSize()

            sitk.WriteImage(image, output_path)

            return {
                'success': True,
                'output_path': output_path,
                'spacing': spacing,
                'size': size
            }

        except Exception as e:
            print(f"Error converting series to NIfTI: {e}")
            return {'success': False, 'error': str(e)}

    def load_nifti_image(self, nifti_path):
        """Load NIfTI image and return image data and header."""
        try:
            nifti_img = nib.load(nifti_path)
            image_data = nifti_img.get_fdata()
            header = nifti_img.header
            return image_data, header
        except Exception as e:
            print(f"Error loading NIfTI image: {e}")
            return None, None

    def basic_brain_segmentation(self, image_data, header):
        """Basic brain segmentation using intensity thresholding."""
        # This is a simplified approach - in practice you'd use trained models
        volumes = {}

        # Get voxel volume for volume calculations
        spacing = header.get_zooms()
        voxel_volume = np.prod(spacing)  # mm³ per voxel

        # Normalize image intensity
        image_norm = (image_data - image_data.min()) / (image_data.max() - image_data.min())

        # Basic thresholding for different structures
        # These are simplified heuristics - real segmentation would use trained models

        # Brain mask (remove background)
        brain_mask = image_norm > 0.1
        total_brain_volume = np.sum(brain_mask) * voxel_volume

        # Estimate CSF (cerebrospinal fluid) - typically darker regions within brain
        csf_mask = (image_norm > 0.1) & (image_norm < 0.3) & brain_mask
        csf_volume = np.sum(csf_mask) * voxel_volume

        # Estimate gray matter
        gray_matter_mask = (image_norm > 0.3) & (image_norm < 0.7) & brain_mask
        gray_matter_volume = np.sum(gray_matter_mask) * voxel_volume

        # Estimate white matter
        white_matter_mask = (image_norm > 0.7) & brain_mask
        white_matter_volume = np.sum(white_matter_mask) * voxel_volume

        # Approximate regional volumes based on anatomical expectations
        # These are rough estimates - real segmentation would be more accurate
        volumes = {
            'Total-Brain': int(total_brain_volume),
            'CSF': int(csf_volume),
            'Gray-Matter': int(gray_matter_volume),
            'White-Matter': int(white_matter_volume),
            'Left-Hippocampus': int(gray_matter_volume * 0.005),  # ~0.5% of gray matter
            'Right-Hippocampus': int(gray_matter_volume * 0.005),
            'Left-Lateral-Ventricle': int(csf_volume * 0.3),  # ~30% of CSF
            'Right-Lateral-Ventricle': int(csf_volume * 0.3),
            'Left-Frontal-Pole': int(gray_matter_volume * 0.08),  # ~8% of gray matter
            'Right-Frontal-Pole': int(gray_matter_volume * 0.08),
        }

        return volumes

    def analyze_findings(self, volumes):
        """Analyze brain volumes and derive clinical findings."""
        findings = {}

        # Hippocampal analysis
        left_hippo = volumes.get('Left-Hippocampus', 0)
        right_hippo = volumes.get('Right-Hippocampus', 0)
        total_hippo = left_hippo + right_hippo

        findings['hippocampal_assessment'] = {
            'left_volume_mm3': left_hippo,
            'right_volume_mm3': right_hippo,
            'total_volume_mm3': total_hippo,
            'normal_range': '5600-8000 mm³ (total)',
            'atrophy_present': total_hippo < 5600,
            'severity': self.get_severity(total_hippo, 5600, 4800, 4000),
            'asymmetry': abs(left_hippo - right_hippo) / max(left_hippo, right_hippo) > 0.15
        }

        # Ventricular analysis
        left_vent = volumes.get('Left-Lateral-Ventricle', 0)
        right_vent = volumes.get('Right-Lateral-Ventricle', 0)
        total_vent = left_vent + right_vent

        findings['ventricular_assessment'] = {
            'left_volume_mm3': left_vent,
            'right_volume_mm3': right_vent,
            'total_volume_mm3': total_vent,
            'normal_range': '14000-30000 mm³ (total)',
            'enlargement_present': total_vent > 30000,
            'severity': self.get_severity(total_vent, 30000, 40000, 50000, inverse=True),
            'asymmetry': abs(left_vent - right_vent) / max(left_vent, right_vent) > 0.20
        }

        # Frontal lobe analysis
        left_frontal = volumes.get('Left-Frontal-Pole', 0)
        right_frontal = volumes.get('Right-Frontal-Pole', 0)
        total_frontal = left_frontal + right_frontal

        findings['frontal_assessment'] = {
            'left_volume_mm3': left_frontal,
            'right_volume_mm3': right_frontal,
            'total_volume_mm3': total_frontal,
            'normal_range': '6400-10000 mm³ (total)',
            'atrophy_present': total_frontal < 6400,
            'severity': self.get_severity(total_frontal, 6400, 5600, 4800),
            'asymmetry': abs(left_frontal - right_frontal) / max(left_frontal, right_frontal) > 0.15
        }

        # Overall brain analysis
        total_brain = volumes.get('Total-Brain', 0)
        findings['overall_brain'] = {
            'total_volume_mm3': total_brain,
            'normal_range': '1200000-1600000 mm³',
            'atrophy_present': total_brain < 1200000,
            'severity': self.get_severity(total_brain, 1200000, 1100000, 1000000)
        }

        # Generate summary
        findings['clinical_summary'] = self.generate_clinical_summary(findings)

        return findings

    def get_severity(self, value, threshold1, threshold2, threshold3, inverse=False):
        """Determine severity based on thresholds."""
        if inverse:  # For conditions where higher values are worse (like ventriculomegaly)
            if value <= threshold1:
                return 'normal'
            elif value <= threshold2:
                return 'mild'
            elif value <= threshold3:
                return 'moderate'
            else:
                return 'severe'
        else:  # For conditions where lower values are worse (like atrophy)
            if value >= threshold1:
                return 'normal'
            elif value >= threshold2:
                return 'mild'
            elif value >= threshold3:
                return 'moderate'
            else:
                return 'severe'

    def generate_clinical_summary(self, findings):
        """Generate a clinical summary based on findings."""
        summary = []

        # Check each assessment
        for assessment_name, assessment_data in findings.items():
            if assessment_name == 'clinical_summary':
                continue

            if assessment_data.get('atrophy_present') or assessment_data.get('enlargement_present'):
                severity = assessment_data.get('severity', 'unknown')
                condition = assessment_name.replace('_assessment', '').replace('_', ' ')

                if assessment_data.get('atrophy_present'):
                    summary.append(f"{severity} {condition} atrophy")
                elif assessment_data.get('enlargement_present'):
                    summary.append(f"{severity} {condition} enlargement")

            # Check for asymmetry
            if assessment_data.get('asymmetry'):
                condition = assessment_name.replace('_assessment', '').replace('_', ' ')
                summary.append(f"{condition} asymmetry")

        if not summary:
            return "No significant abnormalities detected."
        else:
            return "Findings: " + ", ".join(summary) + "."

    def process_dicom_folder(self, dicom_folder, output_dir="./output", process_all_series=True):
        """Complete pipeline: DICOM → NIfTI → Analysis → Findings for all series."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing DICOM folder: {dicom_folder}")

        # Find all DICOM series
        series_list = self.find_dicom_series(dicom_folder)

        if not series_list:
            print("No DICOM series found.")
            return None

        print(f"Found {len(series_list)} DICOM series:")
        for i, series in enumerate(series_list):
            print(f"  {i+1}. {series['series_description']} ({series['num_files']} files)")

        all_results = []

        # Process each series
        for i, series in enumerate(series_list):
            print(f"\nProcessing series {i+1}/{len(series_list)}: {series['series_description']}")

            # Skip if not processing all series and this is not the first/largest series
            if not process_all_series and i > 0:
                # Find the series with most files (likely the main scan)
                largest_series = max(series_list, key=lambda x: x['num_files'])
                if series != largest_series:
                    print(f"  Skipping (not the largest series)")
                    continue

            # Convert series to NIfTI
            series_output_path = os.path.join(output_dir, f"brain_series_{i+1}_{series['series_id']}.nii.gz")

            conversion_result = self.convert_series_to_nifti(series['files'], series_output_path)

            if not conversion_result['success']:
                print(f"  Failed to convert series: {conversion_result['error']}")
                continue

            print(f"  Converted to: {series_output_path}")
            print(f"  Image size: {conversion_result['size']}")
            print(f"  Voxel spacing: {conversion_result['spacing']}")

            # Load NIfTI image
            print("  Loading NIfTI image...")
            image_data, header = self.load_nifti_image(series_output_path)

            if image_data is None:
                print(f"  Failed to load NIfTI image")
                continue

            # Perform basic segmentation and volume analysis
            print("  Performing brain segmentation and volume analysis...")
            volumes = self.basic_brain_segmentation(image_data, header)

            # Analyze findings
            print("  Analyzing clinical findings...")
            findings = self.analyze_findings(volumes)

            # Prepare results for this series
            series_results = {
                'series_number': i + 1,
                'series_id': series['series_id'],
                'series_description': series['series_description'],
                'num_files': series['num_files'],
                'dicom_folder': dicom_folder,
                'nifti_path': series_output_path,
                'image_shape': image_data.shape,
                'voxel_spacing': conversion_result['spacing'],
                'volumes': volumes,
                'findings': findings,
                'timestamp': str(np.datetime64('now'))
            }

            all_results.append(series_results)
            print(f"  ✓ Series {i+1} processed successfully")

        if not all_results:
            print("No series were successfully processed.")
            return None

        # Create combined results
        combined_results = {
            'dicom_folder': dicom_folder,
            'total_series': len(series_list),
            'processed_series': len(all_results),
            'series_results': all_results,
            'timestamp': str(np.datetime64('now'))
        }

        return combined_results

    def process_single_series(self, dicom_folder, series_index=0, output_dir="./output"):
        """Process a single series from a folder (for backward compatibility)."""
        series_list = self.find_dicom_series(dicom_folder)

        if not series_list:
            return None

        if series_index >= len(series_list):
            print(f"Series index {series_index} not found. Available series: {len(series_list)}")
            return None

        # Process only the specified series
        series = series_list[series_index]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing series: {series['series_description']}")

        # Convert to NIfTI
        nifti_path = os.path.join(output_dir, "brain.nii.gz")
        conversion_result = self.convert_series_to_nifti(series['files'], nifti_path)

        if not conversion_result['success']:
            return None

        # Load and analyze
        image_data, header = self.load_nifti_image(nifti_path)
        if image_data is None:
            return None

        volumes = self.basic_brain_segmentation(image_data, header)
        findings = self.analyze_findings(volumes)

        # Return single series result (backward compatible format)
        return {
            'dicom_folder': dicom_folder,
            'nifti_path': nifti_path,
            'image_shape': image_data.shape,
            'voxel_spacing': conversion_result['spacing'],
            'volumes': volumes,
            'findings': findings,
            'timestamp': str(np.datetime64('now'))
        }

    def print_results(self, results):
        """Print formatted results for single or multiple series."""
        if not results:
            print("No results to display.")
            return

        # Check if this is a multi-series result
        if 'series_results' in results:
            self.print_multi_series_results(results)
        else:
            self.print_single_series_results(results)

    def print_single_series_results(self, results):
        """Print results for a single series (backward compatibility)."""
        print("\n" + "="*70)
        print("BRAIN MRI ANALYSIS RESULTS")
        print("="*70)

        print(f"\nInput: {results['dicom_folder']}")
        print(f"NIfTI: {results['nifti_path']}")
        print(f"Image Shape: {results['image_shape']}")
        print(f"Voxel Spacing: {results['voxel_spacing']}")
        print(f"Analysis Time: {results['timestamp']}")

        self._print_volumes_and_findings(results)

    def print_multi_series_results(self, results):
        """Print results for multiple series."""
        print("\n" + "="*70)
        print("BRAIN MRI ANALYSIS RESULTS - MULTIPLE SERIES")
        print("="*70)

        print(f"\nDICOM Folder: {results['dicom_folder']}")
        print(f"Total Series Found: {results['total_series']}")
        print(f"Successfully Processed: {results['processed_series']}")
        print(f"Analysis Time: {results['timestamp']}")

        # Print summary of all series
        print("\n" + "="*70)
        print("SERIES SUMMARY")
        print("="*70)

        for i, series_result in enumerate(results['series_results']):
            print(f"\nSeries {series_result['series_number']}: {series_result['series_description']}")
            print(f"  Files: {series_result['num_files']}")
            print(f"  Shape: {series_result['image_shape']}")
            print(f"  NIfTI: {series_result['nifti_path']}")

            # Print key findings
            clinical_summary = series_result['findings']['clinical_summary']
            print(f"  Findings: {clinical_summary}")

        # Print detailed results for each series
        for i, series_result in enumerate(results['series_results']):
            print(f"\n" + "="*70)
            print(f"DETAILED RESULTS - SERIES {series_result['series_number']}")
            print(f"Description: {series_result['series_description']}")
            print("="*70)

            self._print_volumes_and_findings(series_result)

    def _print_volumes_and_findings(self, results):
        """Print volumes and findings for a single series."""
        print("\n" + "="*70)
        print("BRAIN STRUCTURE VOLUMES")
        print("="*70)
        for structure, volume in results['volumes'].items():
            print(f"{structure:<25}: {volume:>10,} mm³")

        print("\n" + "="*70)
        print("CLINICAL FINDINGS")
        print("="*70)

        for finding_name, finding_data in results['findings'].items():
            if finding_name == 'clinical_summary':
                print(f"\nCLINICAL SUMMARY:")
                print(f"  {finding_data}")
                continue

            print(f"\n{finding_name.replace('_', ' ').title()}:")

            if 'left_volume_mm3' in finding_data:
                print(f"  Left Volume:  {finding_data['left_volume_mm3']:,} mm³")
                print(f"  Right Volume: {finding_data['right_volume_mm3']:,} mm³")
                print(f"  Total Volume: {finding_data['total_volume_mm3']:,} mm³")
            elif 'total_volume_mm3' in finding_data:
                print(f"  Total Volume: {finding_data['total_volume_mm3']:,} mm³")

            print(f"  Normal Range: {finding_data['normal_range']}")
            print(f"  Severity: {finding_data['severity']}")

            if finding_data.get('atrophy_present'):
                print(f"  Status: ATROPHY DETECTED")
            elif finding_data.get('enlargement_present'):
                print(f"  Status: ENLARGEMENT DETECTED")
            else:
                print(f"  Status: NORMAL")

            if finding_data.get('asymmetry'):
                print(f"  Asymmetry: PRESENT")

    def save_results_json(self, results, filename="brain_analysis_results.json"):
        """Save results to JSON file."""
        if results:
            import json

            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            # Clean results for JSON serialization
            clean_results = {}
            for key, value in results.items():
                if key == 'voxel_spacing':
                    clean_results[key] = [float(x) for x in value]
                else:
                    clean_results[key] = value

            with open(filename, "w") as f:
                json.dump(clean_results, f, indent=2, default=convert_numpy)

            print(f"Results saved to {filename}")

    def save_results_txt(self, results, filename="brain_analysis_results.txt"):
        """Save results to a human-readable text file."""
        if not results:
            print("No results to save.")
            return

        with open(filename, "w") as f:
            f.write("="*70 + "\n")
            f.write("BRAIN MRI ANALYSIS RESULTS\n")
            f.write("="*70 + "\n\n")

            # Basic information
            f.write("SCAN INFORMATION:\n")
            f.write("-" * 50 + "\n")
            f.write(f"DICOM Folder: {results['dicom_folder']}\n")
            f.write(f"NIfTI Path: {results['nifti_path']}\n")
            f.write(f"Image Shape: {results['image_shape']}\n")
            f.write(f"Voxel Spacing: {results['voxel_spacing']}\n")
            f.write(f"Analysis Time: {results['timestamp']}\n\n")

            # Volume measurements
            f.write("BRAIN STRUCTURE VOLUMES:\n")
            f.write("-" * 50 + "\n")
            for structure, volume in results['volumes'].items():
                f.write(f"{structure:<25}: {volume:>10,} mm³\n")
            f.write("\n")

            # Clinical findings
            f.write("CLINICAL FINDINGS:\n")
            f.write("-" * 50 + "\n")

            for finding_name, finding_data in results['findings'].items():
                if finding_name == 'clinical_summary':
                    f.write(f"\nCLINICAL SUMMARY:\n")
                    f.write(f"  {finding_data}\n")
                    continue

                f.write(f"\n{finding_name.replace('_', ' ').title()}:\n")

                if 'left_volume_mm3' in finding_data:
                    f.write(f"  Left Volume:  {finding_data['left_volume_mm3']:,} mm³\n")
                    f.write(f"  Right Volume: {finding_data['right_volume_mm3']:,} mm³\n")
                    f.write(f"  Total Volume: {finding_data['total_volume_mm3']:,} mm³\n")
                elif 'total_volume_mm3' in finding_data:
                    f.write(f"  Total Volume: {finding_data['total_volume_mm3']:,} mm³\n")

                f.write(f"  Normal Range: {finding_data['normal_range']}\n")
                f.write(f"  Severity: {finding_data['severity']}\n")

                if finding_data.get('atrophy_present'):
                    f.write(f"  Status: ATROPHY DETECTED\n")
                elif finding_data.get('enlargement_present'):
                    f.write(f"  Status: ENLARGEMENT DETECTED\n")
                else:
                    f.write(f"  Status: NORMAL\n")

                if finding_data.get('asymmetry'):
                    f.write(f"  Asymmetry: PRESENT\n")

            # Add detailed report section
            f.write("\n" + "="*70 + "\n")
            f.write("DETAILED CLINICAL REPORT\n")
            f.write("="*70 + "\n\n")

            # Generate detailed report
            self._write_detailed_report(f, results)

            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")

        print(f"Results saved to {filename}")

    def _write_detailed_report(self, f, results):
        """Write a detailed clinical report."""
        findings = results['findings']

        f.write("IMPRESSION:\n")
        f.write(f"{findings['clinical_summary']}\n\n")

        f.write("DETAILED FINDINGS:\n\n")

        # Hippocampal findings
        hippo = findings.get('hippocampal_assessment', {})
        f.write("1. HIPPOCAMPAL VOLUME:\n")
        f.write(f"   - Left hippocampus: {hippo.get('left_volume_mm3', 'N/A'):,} mm³\n")
        f.write(f"   - Right hippocampus: {hippo.get('right_volume_mm3', 'N/A'):,} mm³\n")
        f.write(f"   - Total hippocampal volume: {hippo.get('total_volume_mm3', 'N/A'):,} mm³\n")
        f.write(f"   - Assessment: {hippo.get('severity', 'N/A')} {'atrophy' if hippo.get('atrophy_present') else 'volume'}\n")
        if hippo.get('asymmetry'):
            f.write(f"   - Asymmetry: Present (>15% difference between sides)\n")
        f.write(f"   - Reference range: {hippo.get('normal_range', 'N/A')}\n\n")

        # Ventricular findings
        vent = findings.get('ventricular_assessment', {})
        f.write("2. VENTRICULAR VOLUME:\n")
        f.write(f"   - Left lateral ventricle: {vent.get('left_volume_mm3', 'N/A'):,} mm³\n")
        f.write(f"   - Right lateral ventricle: {vent.get('right_volume_mm3', 'N/A'):,} mm³\n")
        f.write(f"   - Total ventricular volume: {vent.get('total_volume_mm3', 'N/A'):,} mm³\n")
        f.write(f"   - Assessment: {vent.get('severity', 'N/A')} {'enlargement' if vent.get('enlargement_present') else 'size'}\n")
        if vent.get('asymmetry'):
            f.write(f"   - Asymmetry: Present (>20% difference between sides)\n")
        f.write(f"   - Reference range: {vent.get('normal_range', 'N/A')}\n\n")

        # Frontal lobe findings
        frontal = findings.get('frontal_assessment', {})
        f.write("3. FRONTAL LOBE VOLUME:\n")
        f.write(f"   - Left frontal pole: {frontal.get('left_volume_mm3', 'N/A'):,} mm³\n")
        f.write(f"   - Right frontal pole: {frontal.get('right_volume_mm3', 'N/A'):,} mm³\n")
        f.write(f"   - Total frontal volume: {frontal.get('total_volume_mm3', 'N/A'):,} mm³\n")
        f.write(f"   - Assessment: {frontal.get('severity', 'N/A')} {'atrophy' if frontal.get('atrophy_present') else 'volume'}\n")
        if frontal.get('asymmetry'):
            f.write(f"   - Asymmetry: Present (>15% difference between sides)\n")
        f.write(f"   - Reference range: {frontal.get('normal_range', 'N/A')}\n\n")

        # Overall brain findings
        overall = findings.get('overall_brain', {})
        f.write("4. OVERALL BRAIN VOLUME:\n")
        f.write(f"   - Total brain volume: {overall.get('total_volume_mm3', 'N/A'):,} mm³\n")
        f.write(f"   - Assessment: {overall.get('severity', 'N/A')} {'atrophy' if overall.get('atrophy_present') else 'volume'}\n")
        f.write(f"   - Reference range: {overall.get('normal_range', 'N/A')}\n\n")

        # Technical notes
        f.write("TECHNICAL NOTES:\n")
        f.write("- Volume measurements calculated using basic intensity thresholding\n")
        f.write("- Results are estimates and should be interpreted by qualified radiologists\n")
        f.write("- Normal ranges are approximate and may vary by population\n")
        f.write("- For clinical diagnosis, correlation with symptoms and other imaging is essential\n")

    def save_results(self, results, format="both", json_filename="brain_analysis_results.json", txt_filename="brain_analysis_results.txt"):
        """Save results in specified format(s) - handles both single and multi-series."""
        if not results:
            print("No results to save.")
            return

        # Check if this is multi-series results
        if 'series_results' in results:
            # Save multi-series results
            if format.lower() in ["json", "both"]:
                self.save_results_json(results, json_filename)

            if format.lower() in ["txt", "text", "both"]:
                self.save_multi_series_txt(results, txt_filename)

            # Also save individual series results
            for series_result in results['series_results']:
                series_num = series_result['series_number']
                series_desc = series_result['series_description'].replace(' ', '_').replace('/', '_')

                if format.lower() in ["json", "both"]:
                    json_name = f"series_{series_num}_{series_desc}.json"
                    self.save_results_json(series_result, json_name)

                if format.lower() in ["txt", "text", "both"]:
                    txt_name = f"series_{series_num}_{series_desc}.txt"
                    self.save_results_txt(series_result, txt_name)
        else:
            # Save single series results (backward compatibility)
            if format.lower() in ["json", "both"]:
                self.save_results_json(results, json_filename)

            if format.lower() in ["txt", "text", "both"]:
                self.save_results_txt(results, txt_filename)

        if format.lower() == "both":
            print(f"Results saved in both formats")

    def save_multi_series_txt(self, results, filename="multi_series_brain_analysis.txt"):
        """Save multi-series results to a comprehensive text file."""
        with open(filename, "w") as f:
            f.write("="*70 + "\n")
            f.write("BRAIN MRI ANALYSIS RESULTS - MULTIPLE SERIES\n")
            f.write("="*70 + "\n\n")

            # Overall information
            f.write("SCAN INFORMATION:\n")
            f.write("-" * 50 + "\n")
            f.write(f"DICOM Folder: {results['dicom_folder']}\n")
            f.write(f"Total Series Found: {results['total_series']}\n")
            f.write(f"Successfully Processed: {results['processed_series']}\n")
            f.write(f"Analysis Time: {results['timestamp']}\n\n")

            # Series summary
            f.write("SERIES SUMMARY:\n")
            f.write("-" * 50 + "\n")
            for series_result in results['series_results']:
                f.write(f"Series {series_result['series_number']}: {series_result['series_description']}\n")
                f.write(f"  Files: {series_result['num_files']}\n")
                f.write(f"  Shape: {series_result['image_shape']}\n")
                f.write(f"  Findings: {series_result['findings']['clinical_summary']}\n")
                f.write(f"  NIfTI: {series_result['nifti_path']}\n\n")

            # Detailed results for each series
            for i, series_result in enumerate(results['series_results']):
                f.write("="*70 + "\n")
                f.write(f"DETAILED RESULTS - SERIES {series_result['series_number']}\n")
                f.write(f"Description: {series_result['series_description']}\n")
                f.write("="*70 + "\n\n")

                # Write detailed findings for this series
                self._write_series_details(f, series_result)

                if i < len(results['series_results']) - 1:
                    f.write("\n")

            f.write("\n" + "="*70 + "\n")
            f.write("END OF MULTI-SERIES REPORT\n")
            f.write("="*70 + "\n")

        print(f"Multi-series results saved to {filename}")

    def _write_series_details(self, f, series_result):
        """Write detailed analysis for a single series."""
        # Volume measurements
        f.write("BRAIN STRUCTURE VOLUMES:\n")
        f.write("-" * 50 + "\n")
        for structure, volume in series_result['volumes'].items():
            f.write(f"{structure:<25}: {volume:>10,} mm³\n")
        f.write("\n")

        # Clinical findings
        f.write("CLINICAL FINDINGS:\n")
        f.write("-" * 50 + "\n")

        for finding_name, finding_data in series_result['findings'].items():
            if finding_name == 'clinical_summary':
                f.write(f"\nCLINICAL SUMMARY:\n")
                f.write(f"  {finding_data}\n")
                continue

            f.write(f"\n{finding_name.replace('_', ' ').title()}:\n")

            if 'left_volume_mm3' in finding_data:
                f.write(f"  Left Volume:  {finding_data['left_volume_mm3']:,} mm³\n")
                f.write(f"  Right Volume: {finding_data['right_volume_mm3']:,} mm³\n")
                f.write(f"  Total Volume: {finding_data['total_volume_mm3']:,} mm³\n")
            elif 'total_volume_mm3' in finding_data:
                f.write(f"  Total Volume: {finding_data['total_volume_mm3']:,} mm³\n")

            f.write(f"  Normal Range: {finding_data['normal_range']}\n")
            f.write(f"  Severity: {finding_data['severity']}\n")

            if finding_data.get('atrophy_present'):
                f.write(f"  Status: ATROPHY DETECTED\n")
            elif finding_data.get('enlargement_present'):
                f.write(f"  Status: ENLARGEMENT DETECTED\n")
            else:
                f.write(f"  Status: NORMAL\n")

            if finding_data.get('asymmetry'):
                f.write(f"  Asymmetry: PRESENT\n")
