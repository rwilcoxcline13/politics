#!/usr/bin/env python3
"""
Interactive Pinecone Data Updater

PURPOSE:
This script allows users to interactively update Pinecone data using local datasets
stored in the "data" folder. It provides a user-friendly interface to:
- Select datasets from the data directory
- Choose matching variables and target columns
- Update Pinecone records with new data
- Verify changes using the dataset inspector

KEY FEATURES:
- Interactive prompts for dataset and column selection
- Preview mode: Shows match/update rates before making any changes
- Uses existing search_voter functionality to find matching records
- Graceful handling of missing matches and data format issues
- Automatic verification of updates using pinecone_dataset_inspector
- Progress tracking and error reporting

REQUIREMENTS:
- Environment variables: PINECONE_API_KEY, OPENAI_API_KEY
- Data files in the 'data' directory (CSV or Excel formats)
- Existing Pinecone index with voter embeddings

USAGE:
    python interactive_pinecone_updater.py

DEPENDENCIES:
- search_voter (VoterSearcher class)
- pinecone_dataset_inspector (for verification)
- pandas (data processing)
- os, glob (file operations)
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from datetime import datetime
from dotenv import load_dotenv
import re
import string
import asyncio
import concurrent.futures
from threading import Lock

# Load environment variables from .env file
load_dotenv()

# Import our existing modules
from search_voter import VoterSearcher
from pinecone import Pinecone


class PineconeDataUpdater:
    """Interactive Pinecone data updater using local datasets"""
    
    def __init__(self):
        """Initialize the updater with Pinecone connection and voter searcher"""
        try:
            # Initialize Pinecone connection
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
            
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(host="https://quincy-voters-3d2397f.svc.aped-4627-b74a.pinecone.io")
            self.namespace = "voter-names"
            
            # Initialize voter searcher
            self.voter_searcher = VoterSearcher()
            
            print("✓ Successfully connected to Pinecone and initialized voter searcher")
            
        except Exception as e:
            print(f"❌ Error initializing: {e}")
            sys.exit(1)
    
    def list_data_files(self) -> List[str]:
        """List available data files in the data directory"""
        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"❌ Data directory '{data_dir}' not found")
            return []
        
        # Look for CSV and Excel files
        patterns = [
            os.path.join(data_dir, "*.csv"),
            os.path.join(data_dir, "*.xlsx"),
            os.path.join(data_dir, "*.xls")
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        
        return [os.path.basename(f) for f in files]
    
    def select_dataset(self) -> Optional[str]:
        """Interactive dataset selection"""
        files = self.list_data_files()
        
        if not files:
            print("❌ No data files found in the 'data' directory")
            print("   Supported formats: CSV (.csv), Excel (.xlsx, .xls)")
            return None
        
        print("\n📂 Available datasets:")
        for i, file in enumerate(files, 1):
            print(f"   {i}. {file}")
        
        while True:
            try:
                choice = input(f"\nSelect dataset (1-{len(files)}): ").strip()
                idx = int(choice) - 1
                
                if 0 <= idx < len(files):
                    selected_file = files[idx]
                    print(f"✓ Selected: {selected_file}")
                    return os.path.join("data", selected_file)
                else:
                    print(f"❌ Please enter a number between 1 and {len(files)}")
            
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Cancelled by user")
                return None
    
    def load_dataset(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load dataset from file"""
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                print(f"❌ Unsupported file format: {file_path}")
                return None
            
            print(f"✓ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def select_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[List[str]]]:
        """Interactive column selection for matching and updating"""
        print(f"\n📊 Dataset columns ({len(df.columns)} total):")
        columns = list(df.columns)
        
        for i, col in enumerate(columns, 1):
            sample_values = df[col].dropna().head(3).tolist()
            sample_str = ", ".join([str(v)[:30] + "..." if len(str(v)) > 30 else str(v) for v in sample_values])
            print(f"   {i:2d}. {col:<30} | Sample: {sample_str}")
        
        # Select matching column
        print(f"\n🔍 Which column should be used to match records?")
        print("   (This will be used to find corresponding voters in Pinecone)")
        
        while True:
            try:
                choice = input(f"Select matching column (1-{len(columns)}): ").strip()
                idx = int(choice) - 1
                
                if 0 <= idx < len(columns):
                    match_column = columns[idx]
                    print(f"✓ Match column: {match_column}")
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(columns)}")
            
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Cancelled by user")
                return None, None
        
        # Select update columns
        print(f"\n📝 Which column(s) do you want to add/update in Pinecone?")
        print("   (Enter multiple numbers separated by commas, e.g., '2,5,7')")
        
        while True:
            try:
                choices = input("Select update columns: ").strip()
                if not choices:
                    print("❌ Please select at least one column")
                    continue
                
                indices = [int(x.strip()) - 1 for x in choices.split(',')]
                
                if all(0 <= idx < len(columns) for idx in indices):
                    update_columns = [columns[idx] for idx in indices]
                    print(f"✓ Update columns: {', '.join(update_columns)}")
                    return match_column, update_columns
                else:
                    print(f"❌ Please enter numbers between 1 and {len(columns)}")
            
            except ValueError:
                print("❌ Please enter valid numbers separated by commas")
            except KeyboardInterrupt:
                print("\n👋 Cancelled by user")
                return None, None
    
    def normalize_name(self, name: str) -> str:
        """Normalize name for better matching: handle 'LAST, FIRST' format, lowercase, remove punctuation"""
        if not name or pd.isna(name):
            return ""
        
        name_str = str(name).strip()
        if not name_str:
            return ""
        
        # Handle "LAST, FIRST" format for names (if contains comma and reasonable length)
        if ',' in name_str and len(name_str) <= 20:
            parts = name_str.split(',', 1)  # Split on first comma only
            if len(parts) == 2:
                last_part = parts[0].strip()  # Everything before comma
                first_part = parts[1].strip()  # Everything after comma
                # Reorder to "first last" format
                name_str = f"{first_part} {last_part}"
        
        # Convert to lowercase
        name_str = name_str.lower()
        
        # Remove punctuation except spaces and hyphens (keep hyphens for hyphenated names)
        name_str = re.sub(r'[^\w\s\-]', '', name_str)
        
        # Normalize multiple spaces to single space
        name_str = ' '.join(name_str.split())
        
        return name_str
    
    def find_voter_by_name(self, name: str) -> Optional[Dict]:
        """Find voter in Pinecone using the search_voter functionality"""
        try:
            # Normalize and validate name
            normalized_name = self.normalize_name(name)
            if not normalized_name:
                return None
            
            # Use the existing search functionality
            matches = self.voter_searcher.search_enhanced(
                normalized_name, 
                top_k=1, 
                confidence_threshold=0.6
            )
            
            if matches and len(matches) > 0:
                return matches[0]
            
            return None
        
        except Exception as e:
            print(f"⚠️  Error searching for '{name}': {e}")
            return None
    
    def update_pinecone_record(self, voter_id: str, update_data: Dict, current_metadata: Dict = None) -> Tuple[bool, Dict]:
        """Update a single record in Pinecone - optimized version"""
        try:
            # If metadata not provided, fetch it
            if current_metadata is None:
                result = self.index.fetch(ids=[voter_id], namespace=self.namespace)
                if voter_id not in result.vectors:
                    return False, {"error": f"Voter ID {voter_id} not found in Pinecone"}
                
                current_record = result.vectors[voter_id]
                current_metadata = current_record.metadata if hasattr(current_record, 'metadata') else {}
            
            # Update metadata with new data
            updated_metadata = current_metadata.copy()
            actually_updated = {}
            skipped = {}
            
            for key, value in update_data.items():
                # Always update the field (overwrite existing values)
                # Only skip if the new value is exactly the same as existing
                if key in current_metadata and current_metadata.get(key) == value:
                    skipped[key] = current_metadata[key]
                else:
                    updated_metadata[key] = value
                    actually_updated[key] = value
            
            # Only update if there are actual updates
            if actually_updated:
                self.index.update(
                    id=voter_id,
                    set_metadata=updated_metadata,
                    namespace=self.namespace
                )
            
            return True, {"updated": actually_updated, "skipped": skipped}
        
        except Exception as e:
            return False, {"error": str(e)}
    
    def _process_update_batch(self, update_queue: List[Dict], stats: Dict):
        """Process a batch of updates efficiently"""
        for update_item in update_queue:
            try:
                voter_id = update_item['voter_id']
                update_data = update_item['update_data'] 
                original_name = update_item['original_name']
                
                # Update record
                success, result_info = self.update_pinecone_record(voter_id, update_data)
                
                if success:
                    if result_info.get('updated'):
                        stats['updated'] += 1
                        # Successfully updated
                    else:
                        stats['skipped_already_exists'] += 1
                        # No fields were updated (already had same values)
                else:
                    stats['errors'] += 1
                    error_msg = result_info.get('error', 'Unknown error')
                    print(f"❌ Error updating '{original_name}' (ID: {voter_id}): {error_msg}")
            
            except Exception as e:
                stats['errors'] += 1
                print(f"❌ Batch update error for '{update_item.get('original_name', 'Unknown')}': {e}")
    
    def process_dataset(self, df: pd.DataFrame, match_column: str, update_columns: List[str], preview_only: bool = False):
        """Process the entire dataset and update Pinecone records - optimized version with preview mode"""
        mode_text = "preview" if preview_only else "update"
        print(f"\n🚀 Starting {mode_text} process...")
        print(f"   Dataset size: {len(df)} records")
        print(f"   Match column: {match_column}")
        print(f"   Update columns: {', '.join(update_columns)}")
        if not preview_only:
            print(f"   Optimizations: Parallel search (10 threads), batch updates (50 at a time)")
        print(f"   Note: Only errors will be displayed for cleaner output\n")
        
        # Statistics tracking
        stats = {
            'total': len(df),
            'found': 0,
            'updated': 0,
            'skipped_no_match': 0,
            'skipped_no_data': 0,
            'skipped_already_exists': 0,
            'skipped_already_updated': 0,
            'errors': 0
        }
        
        # Track which voter IDs have already been updated (limit one update per person)
        updated_voter_ids = set()
        updated_voter_ids_lock = Lock()
        
        # Pre-process all data for faster execution
        print("📋 Pre-processing data...")
        processed_records = []
        
        for idx, row in df.iterrows():
            original_name = row[match_column]
            normalized_name = self.normalize_name(original_name)
            
            if not normalized_name:
                stats['skipped_no_data'] += 1
                continue
            
            # Prepare update data
            update_data = {}
            for col in update_columns:
                value = row[col]
                if not pd.isna(value):
                    # Convert numpy types to Python types for JSON serialization
                    if isinstance(value, (np.integer, np.int64)):
                        value = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        value = float(value)
                        # Validate float values - reject infinity and NaN
                        if not np.isfinite(value):
                            print(f"⚠️  Skipping invalid value for {col}: {value} (not finite)")
                            continue
                    
                    # Additional validation for string values
                    if isinstance(value, str):
                        value = value.strip()
                        if not value:  # Skip empty strings
                            continue
                    
                    update_data[col] = value
            
            if update_data:
                processed_records.append({
                    'index': idx,
                    'original_name': original_name,
                    'normalized_name': normalized_name,
                    'update_data': update_data
                })
        
        print(f"✓ Pre-processed {len(processed_records)} valid records")
        
        # Process records with parallel search and batch updates
        search_batch_size = 20  # Number of parallel searches
        update_batch_size = 50  # Number of updates to batch together
        
        # Process in parallel batches
        update_queue = []  # Queue for batch updates
        preview_matches = []  # Store matches for preview mode
        
        for batch_start in range(0, len(processed_records), search_batch_size):
            batch_end = min(batch_start + search_batch_size, len(processed_records))
            batch_records = processed_records[batch_start:batch_end]
            
            # Progress indicator
            if batch_start % 100 == 0:
                progress_text = "preview" if preview_only else "processed"
                print(f"📊 Progress: {batch_start}/{len(processed_records)} records {progress_text}...")
            
            # Parallel search for voters
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all searches for this batch
                future_to_record = {
                    executor.submit(self.find_voter_by_name, record['normalized_name']): record 
                    for record in batch_records
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_record):
                    record = future_to_record[future]
                    
                    try:
                        voter_match = future.result()
                        
                        if not voter_match:
                            stats['skipped_no_match'] += 1
                            if preview_only:
                                print(f"❌ No match found: '{record['original_name']}'")
                            continue
                        
                        stats['found'] += 1
                        voter_id = voter_match['voter_id']
                        
                        # Thread-safe check for already updated voters
                        with updated_voter_ids_lock:
                            if voter_id in updated_voter_ids:
                                stats['skipped_already_updated'] += 1
                                if preview_only:
                                    print(f"⏭️  Duplicate: '{record['original_name']}' - voter {voter_id} already in dataset")
                                continue
                            
                            # Mark as being processed
                            updated_voter_ids.add(voter_id)
                        
                        if preview_only:
                            # In preview mode, just count what would be updated
                            preview_matches.append({
                                'voter_id': voter_id,
                                'original_name': record['original_name'],
                                'update_data': record['update_data']
                            })
                            stats['updated'] += 1  # Count as "would be updated"
                        else:
                            # Add to update queue for actual processing
                            update_queue.append({
                                'voter_id': voter_id,
                                'update_data': record['update_data'],
                                'original_name': record['original_name']
                            })
                            
                            # Process batch updates when queue is full
                            if len(update_queue) >= update_batch_size:
                                self._process_update_batch(update_queue, stats)
                                update_queue = []
                    
                    except Exception as e:
                        stats['errors'] += 1
                        print(f"❌ Error searching for '{record['original_name']}': {e}")
        
        # Process remaining updates (only in non-preview mode)
        if not preview_only and update_queue:
            self._process_update_batch(update_queue, stats)
        
        # Print statistics
        if preview_only:
            self.print_preview_statistics(stats, updated_voter_ids, preview_matches)
            return stats, preview_matches
        else:
            self.print_update_statistics(stats, updated_voter_ids)
            return stats, None
    
    def print_update_statistics(self, stats: Dict, updated_voter_ids: set = None):
        """Print final update statistics - enhanced version"""
        print(f"\n📊 Update Complete - Statistics:")
        print(f"   Total records processed: {stats['total']}")
        print(f"   Voters found in Pinecone: {stats['found']}")
        print(f"   Records successfully updated: {stats['updated']}")
        print(f"   Records skipped (no match): {stats['skipped_no_match']}")
        print(f"   Records skipped (no data): {stats['skipped_no_data']}")
        print(f"   Records skipped (already exists): {stats['skipped_already_exists']}")
        print(f"   Records skipped (already updated): {stats['skipped_already_updated']}")
        print(f"   Errors encountered: {stats['errors']}")
        
        if stats['total'] > 0:
            match_rate = (stats['found'] / stats['total']) * 100
            update_rate = (stats['updated'] / stats['total']) * 100
            unique_updates = len(updated_voter_ids) if updated_voter_ids else stats['updated']
            print(f"   Match rate: {match_rate:.1f}%")
            print(f"   Update rate: {update_rate:.1f}%")
            print(f"   Unique voters updated: {unique_updates}")
            
        # Summary
        total_processed = stats['updated'] + stats['skipped_no_match'] + stats['skipped_no_data'] + stats['skipped_already_exists'] + stats['skipped_already_updated'] + stats['errors']
        if total_processed == stats['total']:
            print(f"   ✅ All records processed successfully")
    
    def verify_changes(self):
        """Run the dataset inspector to verify changes"""
        print(f"\n🔍 Verifying changes using dataset inspector...")
        print("   (This will show current Pinecone statistics and sample records)")
        
        try:
            # Import and run the inspector
            import subprocess
            result = subprocess.run([
                sys.executable, "pinecone_dataset_inspector.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Verification complete")
                print(result.stdout)
            else:
                print("⚠️  Inspector completed with warnings:")
                print(result.stderr)
        
        except Exception as e:
            print(f"⚠️  Could not run verification: {e}")
            print("   You can manually run: python pinecone_dataset_inspector.py")
    
    def get_pinecone_total_count(self) -> int:
        """Get the total number of vectors in the Pinecone index"""
        try:
            # Get index statistics
            stats = self.index.describe_index_stats()
            
            # Get total vector count from the namespace
            if 'namespaces' in stats and self.namespace in stats['namespaces']:
                total_count = stats['namespaces'][self.namespace]['vector_count']
                return total_count
            else:
                # Fallback: get total across all namespaces
                return stats.get('total_vector_count', 0)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not get Pinecone total count: {e}")
            return 0

    def print_preview_statistics(self, stats: Dict, unique_voter_ids: set, preview_matches: List[Dict]):
        """Print preview statistics before actual updates"""
        print(f"\n📊 PREVIEW RESULTS - What would happen:")
        print(f"   Total records in dataset: {stats['total']}")
        print(f"   Records with valid data: {stats['total'] - stats['skipped_no_data']}")
        print(f"   Voters found in Pinecone: {stats['found']}")
        print(f"   Records that would be updated: {stats['updated']}")
        print(f"   Records with no match: {stats['skipped_no_match']}")
        print(f"   Records with no valid data: {stats['skipped_no_data']}")
        print(f"   Duplicate voters in dataset: {stats['skipped_already_updated']}")
        print(f"   Search errors: {stats['errors']}")
        
        if stats['total'] > 0:
            # Get total Pinecone dataset size
            pinecone_total = self.get_pinecone_total_count()
            
            match_rate = (stats['found'] / stats['total']) * 100
            update_rate = (stats['updated'] / stats['total']) * 100
            unique_updates = len(unique_voter_ids)
            print(f"\n📈 Projected Rates:")
            print(f"   Match rate: {match_rate:.1f}% ({stats['found']}/{stats['total']})")
            print(f"   Update rate: {update_rate:.1f}% ({stats['updated']}/{stats['total']})")
            print(f"   Unique voters to be updated: {unique_updates}")
            
            # Add Pinecone dataset perspective
            if pinecone_total > 0:
                pinecone_match_rate = (stats['found'] / pinecone_total) * 100
                pinecone_update_rate = (stats['updated'] / pinecone_total) * 100
                print(f"\n📊 Pinecone Dataset Impact:")
                print(f"   Total vectors in Pinecone: {pinecone_total:,}")
                print(f"   Match rate vs Pinecone: {pinecone_match_rate:.2f}% ({stats['found']}/{pinecone_total:,})")
                print(f"   Update rate vs Pinecone: {pinecone_update_rate:.2f}% ({stats['updated']}/{pinecone_total:,})")
                print(f"   Percentage of Pinecone being updated: {pinecone_update_rate:.2f}%")
            
            # Show some example matches
            if preview_matches and len(preview_matches) > 0:
                print(f"\n🔍 Example matches (first 5):")
                for i, match in enumerate(preview_matches[:5]):
                    update_fields = list(match['update_data'].keys())
                    print(f"   {i+1}. '{match['original_name']}' → Voter {match['voter_id']} (fields: {', '.join(update_fields)})")
                
                if len(preview_matches) > 5:
                    print(f"   ... and {len(preview_matches) - 5} more matches")

    def preview_dataset_updates(self, df: pd.DataFrame, match_column: str, update_columns: List[str]):
        """Preview what would happen during the update process without making changes"""
        print(f"\n🔍 PREVIEW MODE - Analyzing potential updates...")
        print("   This will search for matches but won't make any changes to Pinecone.")
        
        # Run process in preview mode
        stats, preview_matches = self.process_dataset(df, match_column, update_columns, preview_only=True)
        
        return stats, preview_matches
    
    def run_interactive_update(self):
        """Main interactive update process with preview mode"""
        print("=" * 70)
        print("🎯 INTERACTIVE PINECONE DATA UPDATER")
        print("=" * 70)
        print("This tool helps you update Pinecone voter data using local datasets.")
        print("You can add new fields or update existing empty fields.")
        print("New: Preview mode shows match/update rates before making changes!")
        print()
        
        # Step 1: Select dataset
        dataset_path = self.select_dataset()
        if not dataset_path:
            return
        
        # Step 2: Load dataset
        df = self.load_dataset(dataset_path)
        if df is None:
            return
        
        # Step 3: Select columns
        match_column, update_columns = self.select_columns(df)
        if not match_column or not update_columns:
            return
        
        # Step 4: Preview what would happen
        print(f"\n🔍 STEP 4: PREVIEW MODE")
        print(f"   Let's see what would happen before making any changes...")
        
        preview_confirm = input("\nRun preview analysis? (yes/no): ").strip().lower()
        if preview_confirm not in ['yes', 'y']:
            print("👋 Update cancelled")
            return
        
        # Run preview
        stats, preview_matches = self.preview_dataset_updates(df, match_column, update_columns)
        
        # Step 5: Confirm before actual processing
        print(f"\n⚠️  FINAL CONFIRMATION REQUIRED")
        print(f"   Dataset: {dataset_path}")
        print(f"   Target: Pinecone namespace '{self.namespace}'")
        print(f"   Confidence threshold: 0.6")
        print(f"   Update mode: Will overwrite existing values (one update per person)")
        print(f"   Performance: Parallel processing with 10 threads + batch updates")
        
        if stats['updated'] == 0:
            print(f"\n❌ No records would be updated. Nothing to do.")
            return
        
        print(f"\n📈 Expected Results:")
        print(f"   • {stats['updated']} records will be updated")
        print(f"   • {stats['found']} total matches found")
        print(f"   • {stats['skipped_no_match']} records will be skipped (no match)")
        
        confirm = input(f"\nProceed with updating {stats['updated']} records? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("👋 Update cancelled")
            return
        
        # Step 6: Process dataset (actual updates)
        print(f"\n🚀 STEP 6: PERFORMING ACTUAL UPDATES")
        self.process_dataset(df, match_column, update_columns, preview_only=False)
        
        # Step 7: Verify changes
        verify = input("\nRun verification check? (yes/no): ").strip().lower()
        if verify in ['yes', 'y']:
            self.verify_changes()
        
        print(f"\n✅ Update process complete!")


def main():
    """Main entry point"""
    try:
        updater = PineconeDataUpdater()
        updater.run_interactive_update()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 