"""
Data Quality Validation Script

Validates the quality of downloaded real satellite imagery to ensure it meets
requirements before training AI models.

Validation checks:
1. Required bands present (B02, B03, B04, B08)
2. Vegetation indices within valid ranges (NDVI, SAVI, EVI, NDWI)
3. Minimum temporal coverage (15 dates)
4. Metadata synthetic flag is false
5. Data completeness and integrity

Usage:
    python scripts/validate_data_quality.py --data-dir data/processed --db-path data/agriflux.db
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import rasterio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager
from src.data_processing.vegetation_indices import VegetationIndexCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_quality_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ImageryValidation:
    """Validation results for a single imagery date."""
    imagery_id: int
    acquisition_date: str
    tile_id: str
    checks: List[ValidationResult]
    overall_passed: bool


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    total_imagery: int
    passed_imagery: int
    failed_imagery: int
    imagery_validations: List[ImageryValidation]
    summary_checks: List[ValidationResult]
    overall_passed: bool


class DataQualityValidator:
    """
    Validates data quality for downloaded satellite imagery.
    
    Implements validation checks according to requirements 8.1-8.5:
    - Required bands present
    - Vegetation index range validation
    - Minimum temporal coverage
    - Metadata synthetic flag verification
    - Data completeness checks
    """
    
    # Valid ranges for vegetation indices
    VALID_RANGES = {
        'NDVI': (-1.0, 1.0),
        'SAVI': (-1.5, 1.5),
        'EVI': (-1.0, 1.0),
        'NDWI': (-1.0, 1.0),
        'GNDVI': (-1.0, 1.0),
        'NDSI': (-1.0, 1.0)
    }
    
    # Required bands for processing
    REQUIRED_BANDS = ['B02', 'B03', 'B04', 'B08']
    
    # Required vegetation indices
    REQUIRED_INDICES = ['NDVI', 'SAVI', 'EVI', 'NDWI']
    
    # Minimum temporal coverage (number of dates)
    MIN_TEMPORAL_COVERAGE = 15
    
    def __init__(self, data_dir: Path, db_path: Path):
        """
        Initialize data quality validator.
        
        Args:
            data_dir: Directory containing processed imagery
            db_path: Path to SQLite database
        """
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.db = DatabaseManager(str(db_path))
        
        logger.info(f"DataQualityValidator initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Database: {self.db_path}")
    
    def validate_all(self) -> ValidationReport:
        """
        Run complete validation on all imagery.
        
        Returns:
            ValidationReport with all validation results
        """
        logger.info("="*80)
        logger.info("Starting Data Quality Validation")
        logger.info("="*80)
        
        # Get all imagery from database
        imagery_list = self.db.list_processed_imagery(limit=1000)
        
        if not imagery_list:
            logger.warning("No imagery found in database")
            return ValidationReport(
                timestamp=datetime.now().isoformat(),
                total_imagery=0,
                passed_imagery=0,
                failed_imagery=0,
                imagery_validations=[],
                summary_checks=[
                    ValidationResult(
                        check_name="Database Check",
                        passed=False,
                        message="No imagery found in database"
                    )
                ],
                overall_passed=False
            )
        
        logger.info(f"Found {len(imagery_list)} imagery records in database")
        
        # Validate each imagery
        imagery_validations = []
        passed_count = 0
        failed_count = 0
        
        for i, imagery_record in enumerate(imagery_list, 1):
            logger.info(f"\nValidating {i}/{len(imagery_list)}: "
                       f"{imagery_record['tile_id']}_{imagery_record['acquisition_date']}")
            
            validation = self.validate_imagery(imagery_record)
            imagery_validations.append(validation)
            
            if validation.overall_passed:
                passed_count += 1
                logger.info(f"  ✓ Passed all checks")
            else:
                failed_count += 1
                logger.warning(f"  ✗ Failed validation")
                # Log failed checks
                for check in validation.checks:
                    if not check.passed:
                        logger.warning(f"    - {check.check_name}: {check.message}")
        
        # Run summary checks
        logger.info("\n" + "="*80)
        logger.info("Running Summary Checks")
        logger.info("="*80)
        
        summary_checks = self._run_summary_checks(imagery_list, imagery_validations)
        
        # Determine overall pass/fail
        overall_passed = (
            failed_count == 0 and
            all(check.passed for check in summary_checks)
        )
        
        # Create report
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            total_imagery=len(imagery_list),
            passed_imagery=passed_count,
            failed_imagery=failed_count,
            imagery_validations=imagery_validations,
            summary_checks=summary_checks,
            overall_passed=overall_passed
        )
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("Validation Summary")
        logger.info("="*80)
        logger.info(f"Total imagery: {report.total_imagery}")
        logger.info(f"Passed: {report.passed_imagery}")
        logger.info(f"Failed: {report.failed_imagery}")
        logger.info(f"Overall: {'✓ PASSED' if overall_passed else '✗ FAILED'}")
        logger.info("="*80)
        
        return report
    
    def validate_imagery(self, imagery_record: Dict[str, Any]) -> ImageryValidation:
        """
        Validate a single imagery record.
        
        Args:
            imagery_record: Database record for imagery
            
        Returns:
            ImageryValidation with check results
        """
        checks = []
        
        # Parse metadata
        metadata = self._parse_metadata(imagery_record)
        
        # Check 1: Metadata synthetic flag is false (Requirement 8.4)
        checks.append(self._check_synthetic_flag(metadata))
        
        # Check 2: Required bands present (Requirement 8.1)
        checks.append(self._check_required_bands(imagery_record, metadata))
        
        # Check 3: Vegetation indices within valid ranges (Requirement 8.2)
        checks.append(self._check_index_ranges(imagery_record))
        
        # Check 4: Data file integrity
        checks.append(self._check_file_integrity(imagery_record))
        
        # Determine overall pass/fail for this imagery
        overall_passed = all(check.passed for check in checks)
        
        return ImageryValidation(
            imagery_id=imagery_record['id'],
            acquisition_date=imagery_record['acquisition_date'],
            tile_id=imagery_record['tile_id'],
            checks=checks,
            overall_passed=overall_passed
        )
    
    def _parse_metadata(self, imagery_record: Dict[str, Any]) -> Dict[str, Any]:
        """Parse metadata JSON from imagery record."""
        metadata_json = imagery_record.get('metadata_json', '{}')
        try:
            return json.loads(metadata_json)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse metadata for imagery {imagery_record['id']}")
            return {}
    
    def _check_synthetic_flag(self, metadata: Dict[str, Any]) -> ValidationResult:
        """
        Check that synthetic flag is false for real data.
        
        Validates Requirement 8.4: Check metadata synthetic flag is false
        """
        synthetic = metadata.get('synthetic', True)
        
        if synthetic is False:
            return ValidationResult(
                check_name="Synthetic Flag",
                passed=True,
                message="Data correctly marked as real (synthetic=false)",
                details={'synthetic': False}
            )
        else:
            return ValidationResult(
                check_name="Synthetic Flag",
                passed=False,
                message=f"Data incorrectly marked as synthetic (synthetic={synthetic})",
                details={'synthetic': synthetic}
            )
    
    def _check_required_bands(
        self,
        imagery_record: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """
        Check that all required bands are present.
        
        Validates Requirement 8.1: Check that all required bands are present
        """
        # Check metadata for bands list
        bands_in_metadata = metadata.get('bands', [])
        
        missing_bands = [
            band for band in self.REQUIRED_BANDS
            if band not in bands_in_metadata
        ]
        
        if not missing_bands:
            return ValidationResult(
                check_name="Required Bands",
                passed=True,
                message=f"All required bands present: {', '.join(self.REQUIRED_BANDS)}",
                details={
                    'required': self.REQUIRED_BANDS,
                    'present': bands_in_metadata
                }
            )
        else:
            return ValidationResult(
                check_name="Required Bands",
                passed=False,
                message=f"Missing required bands: {', '.join(missing_bands)}",
                details={
                    'required': self.REQUIRED_BANDS,
                    'present': bands_in_metadata,
                    'missing': missing_bands
                }
            )
    
    def _check_index_ranges(self, imagery_record: Dict[str, Any]) -> ValidationResult:
        """
        Check that vegetation indices are within valid ranges.
        
        Validates Requirement 8.2: Verify vegetation indices are within valid ranges
        """
        issues = []
        index_stats = {}
        
        for index_name in self.REQUIRED_INDICES:
            # Get path from database record
            path_key = f"{index_name.lower()}_path"
            index_path = imagery_record.get(path_key)
            
            if not index_path:
                issues.append(f"{index_name}: path not found in database")
                continue
            
            index_path = Path(index_path)
            
            # Try to load the index data
            try:
                if index_path.exists() and index_path.suffix == '.tif':
                    # Load from GeoTIFF
                    with rasterio.open(index_path) as src:
                        data = src.read(1)
                else:
                    # Try numpy file
                    npy_path = index_path.with_suffix('.npy')
                    if npy_path.exists():
                        data = np.load(npy_path)
                    else:
                        issues.append(f"{index_name}: file not found at {index_path}")
                        continue
                
                # Get valid data (exclude NaN and inf)
                valid_data = data[np.isfinite(data)]
                
                if len(valid_data) == 0:
                    issues.append(f"{index_name}: no valid data")
                    continue
                
                # Check range
                min_val = float(np.min(valid_data))
                max_val = float(np.max(valid_data))
                expected_min, expected_max = self.VALID_RANGES[index_name]
                
                index_stats[index_name] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': float(np.mean(valid_data)),
                    'expected_range': [expected_min, expected_max]
                }
                
                # Allow small tolerance for floating point errors
                tolerance = 0.01
                if min_val < (expected_min - tolerance) or max_val > (expected_max + tolerance):
                    issues.append(
                        f"{index_name}: values outside valid range "
                        f"[{expected_min}, {expected_max}], "
                        f"got [{min_val:.3f}, {max_val:.3f}]"
                    )
            
            except Exception as e:
                issues.append(f"{index_name}: error reading file - {str(e)}")
        
        if not issues:
            return ValidationResult(
                check_name="Index Ranges",
                passed=True,
                message="All vegetation indices within valid ranges",
                details=index_stats
            )
        else:
            return ValidationResult(
                check_name="Index Ranges",
                passed=False,
                message=f"Index range issues: {'; '.join(issues)}",
                details=index_stats
            )
    
    def _check_file_integrity(self, imagery_record: Dict[str, Any]) -> ValidationResult:
        """
        Check that data files exist and are readable.
        """
        issues = []
        
        # Check index files
        for index_name in self.REQUIRED_INDICES:
            path_key = f"{index_name.lower()}_path"
            index_path = imagery_record.get(path_key)
            
            if not index_path:
                issues.append(f"{index_name}: no path in database")
                continue
            
            index_path = Path(index_path)
            
            # Check if file exists
            if not index_path.exists():
                # Try numpy alternative
                npy_path = index_path.with_suffix('.npy')
                if not npy_path.exists():
                    issues.append(f"{index_name}: file not found")
        
        if not issues:
            return ValidationResult(
                check_name="File Integrity",
                passed=True,
                message="All data files exist and are accessible"
            )
        else:
            return ValidationResult(
                check_name="File Integrity",
                passed=False,
                message=f"File integrity issues: {'; '.join(issues)}"
            )
    
    def _run_summary_checks(
        self,
        imagery_list: List[Dict[str, Any]],
        imagery_validations: List[ImageryValidation]
    ) -> List[ValidationResult]:
        """
        Run summary-level validation checks.
        
        Args:
            imagery_list: List of all imagery records
            imagery_validations: List of individual imagery validations
            
        Returns:
            List of summary validation results
        """
        summary_checks = []
        
        # Check 1: Minimum temporal coverage (Requirement 8.3)
        real_imagery_count = sum(
            1 for record in imagery_list
            if not json.loads(record.get('metadata_json', '{}')).get('synthetic', True)
        )
        
        if real_imagery_count >= self.MIN_TEMPORAL_COVERAGE:
            summary_checks.append(ValidationResult(
                check_name="Temporal Coverage",
                passed=True,
                message=f"Sufficient temporal coverage: {real_imagery_count} dates "
                       f"(minimum: {self.MIN_TEMPORAL_COVERAGE})",
                details={
                    'count': real_imagery_count,
                    'minimum': self.MIN_TEMPORAL_COVERAGE
                }
            ))
        else:
            summary_checks.append(ValidationResult(
                check_name="Temporal Coverage",
                passed=False,
                message=f"Insufficient temporal coverage: {real_imagery_count} dates "
                       f"(minimum: {self.MIN_TEMPORAL_COVERAGE})",
                details={
                    'count': real_imagery_count,
                    'minimum': self.MIN_TEMPORAL_COVERAGE
                }
            ))
        
        # Check 2: All imagery passed individual checks
        failed_imagery = [v for v in imagery_validations if not v.overall_passed]
        
        if not failed_imagery:
            summary_checks.append(ValidationResult(
                check_name="Individual Imagery Quality",
                passed=True,
                message="All imagery passed individual quality checks"
            ))
        else:
            summary_checks.append(ValidationResult(
                check_name="Individual Imagery Quality",
                passed=False,
                message=f"{len(failed_imagery)} imagery failed individual quality checks",
                details={
                    'failed_count': len(failed_imagery),
                    'failed_ids': [v.imagery_id for v in failed_imagery]
                }
            ))
        
        # Check 3: Data source consistency
        data_sources = set()
        for record in imagery_list:
            metadata = json.loads(record.get('metadata_json', '{}'))
            if not metadata.get('synthetic', True):
                data_sources.add(metadata.get('data_source', 'Unknown'))
        
        if 'Sentinel Hub API' in data_sources:
            summary_checks.append(ValidationResult(
                check_name="Data Source",
                passed=True,
                message="Real data from Sentinel Hub API",
                details={'sources': list(data_sources)}
            ))
        else:
            summary_checks.append(ValidationResult(
                check_name="Data Source",
                passed=False,
                message=f"Unexpected data sources: {', '.join(data_sources)}",
                details={'sources': list(data_sources)}
            ))
        
        return summary_checks
    
    def save_report(self, report: ValidationReport, output_path: Path):
        """
        Save validation report to JSON file.
        
        Args:
            report: ValidationReport to save
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        report_dict = {
            'timestamp': report.timestamp,
            'total_imagery': report.total_imagery,
            'passed_imagery': report.passed_imagery,
            'failed_imagery': report.failed_imagery,
            'overall_passed': report.overall_passed,
            'summary_checks': [asdict(check) for check in report.summary_checks],
            'imagery_validations': [
                {
                    'imagery_id': v.imagery_id,
                    'acquisition_date': v.acquisition_date,
                    'tile_id': v.tile_id,
                    'overall_passed': v.overall_passed,
                    'checks': [asdict(check) for check in v.checks]
                }
                for v in report.imagery_validations
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Validation report saved to: {output_path}")


def main():
    """Main entry point for validation script."""
    parser = argparse.ArgumentParser(
        description='Validate data quality for downloaded satellite imagery'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed imagery (default: data/processed)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/agriflux.db',
        help='Path to SQLite database (default: data/agriflux.db)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for validation report JSON (default: logs/validation_report_TIMESTAMP.json)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create validator
        validator = DataQualityValidator(
            data_dir=Path(args.data_dir),
            db_path=Path(args.db_path)
        )
        
        # Run validation
        report = validator.validate_all()
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path('logs') / f'validation_report_{timestamp}.json'
        
        # Save report
        validator.save_report(report, output_path)
        
        # Exit with appropriate code
        if report.overall_passed:
            logger.info("\n✓ All validation checks PASSED")
            sys.exit(0)
        else:
            logger.error("\n✗ Validation FAILED - see report for details")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
